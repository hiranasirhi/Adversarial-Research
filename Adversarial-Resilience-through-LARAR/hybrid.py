import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ==================== HYBRID LARAR-ADVNN MODEL ====================
class HybridLARAR_ADVNN(nn.Module):
    """
    Enhanced LARAR-ADVNN with improved LVS computation and monitoring.

    ROOT CAUSE OF SAME WEIGHTS BUG:
    - Previously, LVS was computed inside torch.no_grad(), producing a
      detached tensor. When used in the loss, no gradient flowed back into
      layer_weights_raw, so it never updated.
    FIX:
    - compute_enhanced_lvs_differentiable() runs WITH gradients so the
      LVS values stay on the computation graph.
    - compute_enhanced_lvs() (no_grad version) is kept for eval/monitoring only.
    - get_positive_weights() via softplus keeps weights strictly > 0.
    """
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, num_layers=2,
                 beta=0.3, layer_thresholds=None):
        super(HybridLARAR_ADVNN, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.beta = beta
        self.epsilon = 1e-8

        # Network architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.output = nn.Linear(hidden_dim2, 1)

        # Auxiliary classifiers for multi-level supervision
        self.layer_aux_classifiers = nn.ModuleList([
            nn.Linear(hidden_dim1, 1),
            nn.Linear(hidden_dim2, 1)
        ])

        # Unconstrained raw parameter; always access via get_positive_weights()
        # Init: softplus(0.5413) ≈ 1.0 to match original torch.ones() intent
        self.layer_weights_raw = nn.Parameter(torch.full((num_layers,), 0.5413))

        # Thresholds for early detection (monitoring only)
        self.layer_thresholds = layer_thresholds if layer_thresholds else [0.1, 0.1]

        # For eval/monitoring only (no_grad versions)
        self.layer_vulnerability_scores = {}
        self.early_detection_flags = {}

    def get_positive_weights(self):
        """
        Strictly positive layer weights via softplus.
        softplus(x) = log(1 + exp(x)) > 0 for all x.
        Smooth gradient everywhere — no dead zones unlike clamp/abs.
        """
        return positive_weights = 1.0 + F.softplus(self.layer_weights_raw)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        out = self.output(h2)
        aux1 = torch.sigmoid(self.layer_aux_classifiers[0](h1))
        aux2 = torch.sigmoid(self.layer_aux_classifiers[1](h2))
        return out, aux1, aux2, h1, h2  # also return hidden states

    def compute_lvs_differentiable(self, X_clean, X_adv):
        """
        Differentiable LVS — runs WITH gradient tracking so that
        layer_weights_raw actually receives gradients during training.

        Returns a list of LVS tensors (one per layer), still attached
        to the computation graph.
        """
        # Clean forward pass — keep graph
        _, _, _, h1_clean, h2_clean = self(X_clean)

        # Adversarial forward pass — keep graph
        _, _, _, h1_adv, h2_adv = self(X_adv)

        lvs_list = []
        for h_clean, h_adv in [(h1_clean, h1_adv), (h2_clean, h2_adv)]:
            num = torch.norm(h_adv - h_clean, p=2, dim=1)
            den = torch.norm(h_clean, p=2, dim=1) + self.epsilon
            lvs_list.append(torch.mean(num / den))  # scalar tensor, grad attached

        return lvs_list  # [lvs_layer1, lvs_layer2], both differentiable

    def compute_enhanced_lvs(self, X_clean, X_adv):
        """
        No-grad version — for evaluation and monitoring only.
        Does NOT update layer weights. Use compute_lvs_differentiable() in loss.
        """
        with torch.no_grad():
            lvs_list = self.compute_lvs_differentiable(X_clean, X_adv)
            lvs_scores = {}
            for i, lvs in enumerate(lvs_list):
                val = lvs.item()
                lvs_scores[i] = val
                self.layer_vulnerability_scores[i] = val
                self.early_detection_flags[i] = val > self.layer_thresholds[i]
        return lvs_scores


# ==================== COMPOSITE LOSS ====================
def hybrid_larar_loss(model, X_clean, X_adv, y_true,
                      align_coef=1.0, smooth_coef=0.5, aux_coef=0.2):
    """
    Composite loss. Key fix: uses compute_lvs_differentiable() so that
    layer_weights_raw receives real gradients and actually diverges per layer.
    """
    criterion = nn.BCEWithLogitsLoss()

    X_clean_req = X_clean.clone().detach().requires_grad_(True)
    X_adv_req   = X_adv.clone().detach().requires_grad_(True)

    # --- Forward passes (return hidden states too) ---
    y_pred_clean, aux1_clean, aux2_clean, h1_clean, h2_clean = model(X_clean_req)
    y_pred_adv,   aux1_adv,   aux2_adv,   h1_adv,   h2_adv   = model(X_adv_req)

    # 1. Classification loss
    loss_ce = (criterion(y_pred_clean.squeeze(), y_true) +
               criterion(y_pred_adv.squeeze(),   y_true)) / 2

    # 2. Auxiliary losses
    y_true_exp = y_true.unsqueeze(1)
    loss_aux = aux_coef * (
        F.binary_cross_entropy(aux1_clean, y_true_exp) +
        F.binary_cross_entropy(aux2_clean, y_true_exp)
    )

    # 3. Gradient alignment loss
    grad_clean = torch.autograd.grad(y_pred_clean.sum(), X_clean_req,
                                     create_graph=True, retain_graph=True)[0]
    grad_adv   = torch.autograd.grad(y_pred_adv.sum(),   X_adv_req,
                                     create_graph=True, retain_graph=True)[0]
    loss_align = align_coef * F.mse_loss(grad_clean, grad_adv)

    # 4. Feature smoothness loss (last hidden layer)
    loss_smooth = smooth_coef * F.mse_loss(h2_clean, h2_adv)

    # 5. LVS loss — DIFFERENTIABLE: gradients flow into layer_weights_raw
    lvs_list = model.compute_lvs_differentiable(X_clean, X_adv)
    lvs_stack = torch.stack(lvs_list)          # shape [2], grad attached
    pos_weights = model.get_positive_weights() # shape [2], grad attached
    loss_lvs = model.beta * torch.dot(pos_weights, lvs_stack)

    total_loss = loss_ce + loss_aux + loss_align + loss_smooth + loss_lvs

    # Monitoring metrics (detached, no graph needed)
    with torch.no_grad():
        metrics = {
            'ce':     loss_ce.item(),
            'aux':    loss_aux.item(),
            'align':  loss_align.item(),
            'smooth': loss_smooth.item(),
            'lvs':    loss_lvs.item(),
            'lvs_layer1': lvs_list[0].item(),
            'lvs_layer2': lvs_list[1].item(),
            'early_detection_layer1': lvs_list[0].item() > model.layer_thresholds[0],
            'early_detection_layer2': lvs_list[1].item() > model.layer_thresholds[1],
        }

    return total_loss, metrics


# ==================== PGD ATTACK ====================
def pgd_attack(model, X, y, epsilon=0.1, alpha=0.01, num_iter=10):
    """Projected Gradient Descent adversarial attack."""
    X_adv = X.clone().detach().requires_grad_(True)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(num_iter):
        y_pred, _, _, _, _ = model(X_adv)
        loss = criterion(y_pred.squeeze(), y)
        model.zero_grad()
        loss.backward()

        X_adv.data = X_adv + alpha * X_adv.grad.sign()
        eta = torch.clamp(X_adv - X, min=-epsilon, max=epsilon)
        X_adv.data = torch.clamp(X + eta, min=-5, max=5)
        X_adv.grad.zero_()

    return X_adv.detach()


# ==================== TRAINING ====================
def train_hybrid_model(model, train_loader, test_loader, num_epochs=20,
                       lr=0.001, device='cpu', verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    results = []
    lvs_history = {'layer1': [], 'layer2': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_clean = correct_adv = total_samples = 0
        epoch_lvs = {'layer1': [], 'layer2': []}
        epoch_metrics = {'ce': 0, 'aux': 0, 'align': 0, 'smooth': 0, 'lvs': 0}

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()

            X_adv = pgd_attack(model, X_batch, y_batch, epsilon=0.1)

            optimizer.zero_grad()
            loss, metrics = hybrid_larar_loss(model, X_batch, X_adv, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            epoch_lvs['layer1'].append(metrics['lvs_layer1'])
            epoch_lvs['layer2'].append(metrics['lvs_layer2'])

            with torch.no_grad():
                pred_clean = (torch.sigmoid(model(X_batch)[0]) > 0.5).float().squeeze()
                pred_adv   = (torch.sigmoid(model(X_adv)[0])   > 0.5).float().squeeze()
                correct_clean  += (pred_clean == y_batch).sum().item()
                correct_adv    += (pred_adv   == y_batch).sum().item()
                total_samples  += y_batch.size(0)

        for key in epoch_metrics:
            epoch_metrics[key] /= len(train_loader)

        lvs_history['layer1'].append(np.mean(epoch_lvs['layer1']))
        lvs_history['layer2'].append(np.mean(epoch_lvs['layer2']))

        test_metrics = evaluate_model(model, test_loader, device)
        pos_weights  = model.get_positive_weights()

        results.append({
            'epoch':           epoch + 1,
            'train_loss':      total_loss / len(train_loader),
            'train_acc_clean': correct_clean / total_samples,
            'train_acc_adv':   correct_adv   / total_samples,
            **epoch_metrics,
            **test_metrics,
            'lvs_layer1':     lvs_history['layer1'][-1],
            'lvs_layer2':     lvs_history['layer2'][-1],
            'layer_weight_1': pos_weights[0].item(),
            'layer_weight_2': pos_weights[1].item(),
        })

        if verbose:
            r = results[-1]
            print(f"Epoch {epoch+1:02d}: Loss={r['train_loss']:.4f} | "
                  f"Clean={r['train_acc_clean']:.4f} | Adv={r['train_acc_adv']:.4f}")
            print(f"  LVS  L1={r['lvs_layer1']:.4f}  L2={r['lvs_layer2']:.4f}")
            print(f"  Weights (pos) L1={r['layer_weight_1']:.4f}  "
                  f"L2={r['layer_weight_2']:.4f}  "
                  f"[raw: {model.layer_weights_raw.data.cpu().numpy()}]")

    return results, lvs_history


# ==================== EVALUATION ====================
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    y_true, y_pred_clean, y_pred_adv = [], [], []
    total_lvs = {0: 0.0, 1: 0.0}
    num_batches = 0

    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()
        X_adv   = pgd_attack(model, X_batch, y_batch, epsilon=0.1)

        with torch.no_grad():
            y_pred_c = torch.sigmoid(model(X_batch)[0])
            y_pred_a = torch.sigmoid(model(X_adv)[0])
            lvs_scores = model.compute_enhanced_lvs(X_batch, X_adv)
            for k in total_lvs:
                total_lvs[k] += lvs_scores.get(k, 0.0)
            num_batches += 1

        y_true.extend(y_batch.cpu().numpy())
        y_pred_clean.extend((y_pred_c > 0.5).cpu().numpy().squeeze())
        y_pred_adv.extend(  (y_pred_a > 0.5).cpu().numpy().squeeze())

    return {
        'test_acc_clean':       accuracy_score(y_true, y_pred_clean),
        'test_acc_adv':         accuracy_score(y_true, y_pred_adv),
        'test_precision_clean': precision_score(y_true, y_pred_clean, zero_division=0),
        'test_recall_clean':    recall_score(y_true, y_pred_clean, zero_division=0),
        'test_f1_clean':        f1_score(y_true, y_pred_clean, zero_division=0),
        'test_precision_adv':   precision_score(y_true, y_pred_adv, zero_division=0),
        'test_recall_adv':      recall_score(y_true, y_pred_adv, zero_division=0),
        'test_f1_adv':          f1_score(y_true, y_pred_adv, zero_division=0),
        'test_lvs_layer1':      total_lvs[0] / num_batches,
        'test_lvs_layer2':      total_lvs[1] / num_batches,
    }


# ==================== DATA LOADING ====================
def load_and_preprocess_data(file_path):
    print("Loading dataset...")
    if file_path.endswith('.parquet'):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Parquet read failed ({e}), trying CSV...")
            df = pd.read_csv(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File must be .csv or .parquet")

    if 'label' in df.columns:
        target_col = 'label'
    elif 'attack_cat' in df.columns:
        df['label'] = (df['attack_cat'] != 'Normal').astype(int)
        target_col = 'label'
    else:
        raise ValueError("Cannot find target column ('label' or 'attack_cat')")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mode()[0] if col in cat_cols else 0)
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Loaded {len(X)} samples → train={len(X_train)}, test={len(X_test)}")
    return (torch.FloatTensor(X_train), torch.FloatTensor(X_test),
            torch.FloatTensor(y_train), torch.FloatTensor(y_test))


# ==================== VISUALIZATION ====================
def plot_lvs_evolution(lvs_history, results, save_path):
    epochs = [r['epoch'] for r in results]
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs  = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, lvs_history['layer1'], label='LVS Layer 1', marker='o')
    ax1.plot(epochs, lvs_history['layer2'], label='LVS Layer 2', marker='s', linestyle='--')
    ax1.set(xlabel='Epoch', ylabel='Mean LVS', title='Layer Vulnerability Score Evolution')
    ax1.legend(); ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [r['train_acc_clean'] for r in results], label='Train Clean', color='blue')
    ax2.plot(epochs, [r['train_acc_adv']   for r in results], label='Train Adv',   color='red')
    ax2.plot(epochs, [r['test_acc_clean']  for r in results], label='Test Clean',  color='blue',  linestyle='--')
    ax2.plot(epochs, [r['test_acc_adv']    for r in results], label='Test Adv',    color='red',   linestyle='--')
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy Evolution')
    ax2.legend(); ax2.grid(True, linestyle=':', alpha=0.6)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, [r['layer_weight_1'] for r in results], label='Weight L1 (softplus)', marker='o')
    ax3.plot(epochs, [r['layer_weight_2'] for r in results], label='Weight L2 (softplus)', marker='s')
    ax3.set(xlabel='Epoch', ylabel='Weight (positive)', title='Layer Weights — Should Diverge')
    ax3.legend(); ax3.grid(True, linestyle=':', alpha=0.6)

    ax4 = fig.add_subplot(gs[1, 1])
    for comp in ['ce', 'aux', 'align', 'smooth', 'lvs']:
        ax4.plot(epochs, [r[comp] for r in results], label=f'{comp.upper()}')
    ax4.set(xlabel='Epoch', ylabel='Loss', title='Loss Components')
    ax4.legend(fontsize='small'); ax4.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Hybrid LARAR-ADVNN Training Analysis', fontsize=16)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_final_metrics(results, save_path):
    final = results[-1]
    metrics = ['acc', 'precision', 'recall', 'f1']
    clean_v = [final[f'test_{m}_clean'] for m in metrics]
    adv_v   = [final[f'test_{m}_adv']   for m in metrics]
    x, w    = np.arange(len(metrics)), 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - w/2, clean_v, w, label='Clean')
    r2 = ax.bar(x + w/2, adv_v,   w, label='Adversarial')
    ax.set(ylabel='Score', title='Final Test Performance: Clean vs Adversarial')
    ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()

    for rects in [r1, r2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")


def create_training_report(results, lvs_history, model, save_path):
    final = results[-1]
    pos_w = model.get_positive_weights().data.cpu().numpy()
    lines = [
        "=" * 80,
        "           HYBRID LARAR-ADVNN TRAINING REPORT",
        "=" * 80,
        f"Device:          {next(model.parameters()).device}",
        f"Epochs:          {len(results)}",
        f"Input Dim:       {model.input_dim}",
        f"Beta:            {model.beta}",
        "-" * 80,
        "",
        "## FINAL PERFORMANCE",
        f"  Test Clean Acc:  {final['test_acc_clean']:.4f}",
        f"  Test Adv Acc:    {final['test_acc_adv']:.4f}",
        f"  Test Clean F1:   {final['test_f1_clean']:.4f}",
        f"  Test Adv F1:     {final['test_f1_adv']:.4f}",
        "",
        "## LEARNED LAYER WEIGHTS (positive via softplus)",
    ] + [f"  Layer {i+1}: {w:.4f}" for i, w in enumerate(pos_w)] + [
        "",
        "## LVS (final test batch)",
        f"  Layer 1: {final['test_lvs_layer1']:.4f}",
        f"  Layer 2: {final['test_lvs_layer2']:.4f}",
        "",
        "## PER-EPOCH SUMMARY",
    ] + [
        f"  Ep {r['epoch']:02d} | Loss={r['train_loss']:.4f} | "
        f"CleanAcc={r['train_acc_clean']:.4f} | AdvAcc={r['train_acc_adv']:.4f} | "
        f"LVS1={r['lvs_layer1']:.4f} | LVS2={r['lvs_layer2']:.4f} | "
        f"W1={r['layer_weight_1']:.4f} | W2={r['layer_weight_2']:.4f}"
        for r in results
    ] + [
        "",
        "## FINAL LOSS BREAKDOWN",
    ] + [f"  {c.upper()}: {final[c]:.4f}" for c in ['ce', 'aux', 'align', 'smooth', 'lvs']] + [
        "",
        "=" * 80,
    ]
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    file_path = "/home/kali/Downloads/archive/UNSW_NB15_testing-set.parquet"

    try:
        # ❌ ERROR 1: function name misspelled
        X_train, X_test, y_train, y_test = load_and_preprocessdata(file_path)
    except Exception as e:
        print(f"\nFATAL: Could not load data from '{file_path}'\nError: {e}")
        exit()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64, shuffle=False)

    model = HybridLARAR_ADVNN(
        input_dim=X_train.shape[1],
        beta=0.3,
        layer_thresholds=[0.1, 0.1]
    ).to(device)

    print("\n" + "=" * 60)
    print(model)
    print("=" * 60 + "\n")

    # ❌ ERROR 2: wrong function name
    results, lvs_history = train_hybridmodel(
        model, train_loader, test_loader,
        num_epochs=20, device=device
    )

    print("\n" + "=" * 60 + "\nFINAL RESULTS\n" + "=" * 60)
    for k, v in results[-1].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n" + "=" * 60 + "\nFINAL LAYER WEIGHTS (positive)\n" + "=" * 60)
    for i, w in enumerate(model.get_positive_weights()):
        print(f"  Layer {i+1}: {w.item():.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'results':          results,
        'lvs_history':      lvs_history,
        'layer_weights':    model.get_positive_weights().data.cpu().numpy()
    }, 'hybrid_larar_advnn_model.pth')
    print("\nModel saved → hybrid_larar_advnn_model.pth")

    plot_lvs_evolution(lvs_history, results, 'lvs_evolution.png')
    plot_final_metrics(results, 'final_metrics.png')
    create_training_report(results, lvs_history, model, 'training_report.txt')

    print("\n ALL OUTPUTS GENERATED SUCCESSFULLY!")
    print("  1. hybrid_larar_advnn_model.pth")
    print("  2. lvs_evolution.png")
    print("  3. final_metrics.png")
    print("  4. training_report.txt")
