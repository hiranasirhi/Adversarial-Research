""" 
advnn_train.py 
 
Requirements: 
  pip3 install torch pandas scikit-learn pyarrow 
 
Usage: 
  python3 advnn_train.py 
""" 
 
import os 
import math 
import argparse 
from typing import Tuple 
 
import pandas as pd 
import numpy as np  
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
 
# --------------------------- 
# Config / Hyperparameters 
# --------------------------- 
DATA_PATH = "/home/kali/Desktop/UNSW_NB15_testing-set.parquet"  # update as 
needed 
BATCH_SIZE = 256 
LR = 1e-3 
EPOCHS = 30 
H1 = 256 
H2 = 128 
NUM_CLASSES = 2  # update if multiclass 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
# Curriculum adversarial training settings 
EPS_START = 0.0         # starting epsilon 
EPS_MAX = 0.3           # maximum epsilon (L-inf) 
EPS_INCREASE_MODE = "linear"  # "linear" or "exp" (only linear implemented below) 
PGD_STEPS = 7 
PGD_STEP_SIZE = None  # if None, will be set to EPS_MAX / PGD_STEPS 
 
# Composite loss weights 
LAMBDA_GRAD_ALIGN = 1.0    # weight for gradient alignment loss 
LAMBDA_FEATURE_SMOOTH = 1.0  # weight for feature smoothing loss 
 
SEED = 42 
torch.manual_seed(SEED) 
np.random.seed(SEED) 
 
# --------------------------- 
# Data helpers 
# --------------------------- 
def load_and_preprocess(path: str) -> Tuple[np.ndarray, np.ndarray]: 
    """ 
    Loads the UNSW-NB15 dataset (parquet or csv) and preprocesses: 
    - label encode categorical cols: proto, service, state, attack_cat (if 
present) 
    - replace missing values with column name string (as user requested) 
    - replace infinite with 0 
    - drop/ignore non-feature columns if needed 
    - scale numeric features with StandardScaler 
    Returns: X (numpy float32), y (numpy int64) 
 
    """ 
    # load 
    if path.endswith(".parquet"): 
        df = pd.read_parquet(path) 
    else: 
        df = pd.read_csv(path) 
 
    # Ensure label exists 
    if "label" not in df.columns: 
        raise RuntimeError("Dataset must contain 'label' column with integer 
labels") 
 
    # identify categorical columns commonly in UNSW-NB15 
    categorical_cols = [c for c in ["proto", "service", "state", "attack_cat"] 
if c in df.columns] 
 
    # encode categoricals (use LabelEncoder then store as numeric) 
    for col in categorical_cols: 
        le = LabelEncoder() 
        df[col] = df[col].astype(str).fillna(col)  # replace nulls temporarily 
with col name 
        df[col] = le.fit_transform(df[col]) 
 
    # Replace missing values with column name (string) AS REQUESTED 
    # But numeric columns cannot hold strings, so we'll apply: if column is 
numeric, fillna with column name -> that will coerce to string which is bad. 
    # Interpretation: replace missing values with the column NAME (string) — 
we'll implement this by: 
    #   - For numeric columns, fillna with a sentinel value equal to the column 
index (deterministic numeric substitute). 
    #   - For object/string columns, fillna with column name. 
    # This is a practical compromise to keep numeric dtype consistent. 
    for i, col in enumerate(df.columns): 
        if df[col].isnull().any(): 
            if pd.api.types.is_numeric_dtype(df[col]): 
                # fill numeric NaNs with a deterministic numeric sentinel: column 
index 
                df[col].fillna(float(i), inplace=True) 
            else: 
                df[col].fillna(col, inplace=True) 
 
    # Replace infinite with 0 
    df = df.replace([np.inf, -np.inf], 0) 
 
    # Exclude non-feature columns 
    exclude_columns = [c for c in ["attack_cat"] if c in df.columns]  # keep 
'label' as y 
    features = df.drop(columns=exclude_columns + [], errors='ignore') 
 
 
    # Ensure label is last or separate 
    y = features["label"].astype(int).to_numpy() 
    X = features.drop(columns=["label"], errors='ignore') 
 
    # Numeric scaling only on numeric columns 
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist() 
    scaler = StandardScaler() 
    X_numeric = scaler.fit_transform(X[numeric_cols]) 
    X_other = X.drop(columns=numeric_cols, errors='ignore') 
 
    if len(X_other.columns) > 0: 
        # Convert other columns (if any) to numeric via LabelEncoder (shouldn't 
happen after above) 
        for col in X_other.columns: 
            X_other[col] = LabelEncoder().fit_transform(X_other[col].astype(str)) 
        X_other_np = X_other.to_numpy(dtype=np.float32) 
        X_full = np.hstack([X_numeric.astype(np.float32), 
X_other_np.astype(np.float32)]) 
    else: 
        X_full = X_numeric.astype(np.float32) 
 
    return X_full, y.astype(np.int64) 
 
# --------------------------- 
# Model 
# --------------------------- 
class ADVNN(nn.Module): 
    def __init__(self, input_dim: int, h1: int = H1, h2: int = H2, num_classes: 
int = NUM_CLASSES): 
        super().__init__() 
        self.fc1 = nn.Linear(input_dim, h1) 
        self.bn1 = nn.BatchNorm1d(h1) 
        self.fc2 = nn.Linear(h1, h2) 
        self.bn2 = nn.BatchNorm1d(h2) 
        self.fc_out = nn.Linear(h2, num_classes) 
 
    def forward(self, x, return_features: bool = False): 
        # x: [B, D] 
        x = self.fc1(x) 
        x = self.bn1(x) 
        x = F.relu(x) 
        feat1 = x  # first hidden activation 
        x = self.fc2(x) 
        x = self.bn2(x) 
        x = F.relu(x) 
        feat2 = x  # second hidden activation (we'll use this for feature 
smoothing) 
 
        out = self.fc_out(x) 
        if return_features: 
            return out, feat1, feat2 
        return out 
 
# --------------------------- 
# Adversary (PGD) 
# --------------------------- 
def pgd_attack(model, x, y, eps, steps, step_size, device): 
    """ 
    Performs PGD (L-inf) attack on input batch x (torch tensor). 
    Returns adversarial examples (clamped in L-inf ball around x). 
    """ 
    model.eval() 
    x_orig = x.detach().clone() 
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)  # random start 
    x_adv = x_adv.detach().clone() 
    x_adv.requires_grad = True 
 
    for _ in range(steps): 
        outputs = model(x_adv) 
        loss = F.cross_entropy(outputs, y) 
        loss.backward() 
        # gradient sign step 
        grad = x_adv.grad.detach() 
        x_adv = x_adv.detach() + step_size * torch.sign(grad) 
        # project back to L-inf ball 
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps) 
        x_adv = x_adv.detach() 
        x_adv.requires_grad = True 
 
    model.train() 
    return x_adv.detach() 
 
# ---------------------------
# Loss components
# ---------------------------
def gradient_alignment_loss(model, x_clean, x_adv, y):
    """
    Compute gradient alignment loss between gradients of loss wrt inputs for
    clean and adv examples.
    We'll compute gradients of standard CE loss w.r.t input and compute 1 - cosine_similarity.
    """

    # Ensure grads are computed
    x_clean_req = x_clean.detach().clone().requires_grad_(True)
    x_adv_req = x_adv.detach().clone().requires_grad_(True)

    # Clean grads
    out_clean = model(x_clean_req)
  
    loss_clean = F.cross_entropyy(out_clean, y)

    grad_clean = torch.autograd.grad(
        loss_clean, x_clean_req, retain_graph=True, create_graph=False
    )[0]

    # Adv grads
    out_adv = model(x_adv_req)
    loss_adv = F.cross_entropy(out_adv, y)

    grad_adv = torch.autograd.grad(
        loss_adv, x_adv_req, retain_graph=False, create_graph=False
    )[0]

    # flatten
    g1 = grad_clean.view(grad_clean.size(0), -1)
    g2 = grad_adv.view(grad_adv.size(0), -1)

    # cosine similarity per-sample
    cos = F.cosine_similarity(g1, g2, dim=1, eps=1e-8)

    # want gradients aligned -> maximize cos -> loss = 1 - cos (mean)
    loss = torch.mean(1.0 - cos)
    return loss


def feature_smoothing_loss(feat_clean, feat_adv):
    """
    Encourage feature similarity between clean and adversarial inputs.
    feat_* are activation tensors (B, hidden_dim).
    Use mean squared error between features.
    """
    return F.mse_loss(feat_clean, feat_adv)
 
# --------------------------- 
# Training loop 
# --------------------------- 
def train(model, optimizer, train_loader, epoch, eps_curr): 
    model.train() 
    total_loss = 0.0 
    total_correct = 0 
    total = 0 
 
    for X_batch, y_batch in train_loader: 
        X_batch = X_batch.to(DEVICE) 
        y_batch = y_batch.to(DEVICE) 
 
        # Create adversarial examples with current epsilon 
        step_size = PGD_STEP_SIZE if PGD_STEP_SIZE is not None else max(1e-3, 
eps_curr / max(1, PGD_STEPS)) 
        if eps_curr > 0: 
            x_adv = pgd_attack(model, X_batch, y_batch, eps_curr, PGD_STEPS, 
step_size, DEVICE) 
 
        else: 
            x_adv = X_batch.detach().clone() 
 
        # Forward passes (and get features) 
        outputs_clean, feat1_c, feat2_c = model(X_batch, return_features=True) 
        outputs_adv, feat1_a, feat2_a = model(x_adv, return_features=True) 
 
        # Standard cross-entropy on adversarial (or you can mix with clean) 
        ce_loss = F.cross_entropy(outputs_adv, y_batch) 
 
        # gradient alignment loss (between clean and adv) 
        grad_align = gradient_alignment_loss(model, X_batch, x_adv, y_batch) 
 
        # feature smoothing loss (use second hidden layer features) 
        feat_smooth = feature_smoothing_loss(feat2_c, feat2_a) 
 
        loss = ce_loss + LAMBDA_GRAD_ALIGN * grad_align + LAMBDA_FEATURE_SMOOTH 
* feat_smooth 
 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
 
        total_loss += loss.item() * X_batch.size(0) 
        preds = outputs_adv.argmax(dim=1) 
        total_correct += (preds == y_batch).sum().item() 
        total += X_batch.size(0) 
 
    avg_loss = total_loss / total 
    acc = total_correct / total 
    print(f"Epoch {epoch:02d} | eps {eps_curr:.4f} | Train Loss {avg_loss:.4f} | 
Train Acc {acc:.4f}") 
    return avg_loss, acc 
 
def evaluate(model, loader): 
    model.eval() 
    total = 0 
    correct = 0 
    with torch.no_grad(): 
        for X_batch, y_batch in loader: 
            X_batch = X_batch.to(DEVICE) 
            y_batch = y_batch.to(DEVICE) 
            outputs = model(X_batch) 
            preds = outputs.argmax(dim=1) 
            total += X_batch.size(0) 
            correct += (preds == y_batch).sum().item() 
    return correct / total 
 
 
# --------------------------- 
# Main 
# --------------------------- 
def main(): 
    print("Loading and preprocessing data...") 
    X, y = load_and_preprocess(DATA_PATH) 
    print("X shape:", X.shape, "y shape:", y.shape) 
 
    # Train/test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
stratify=y, random_state=SEED) 
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), 
torch.from_numpy(y_train).long()) 
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), 
torch.from_numpy(y_test).long()) 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
drop_last=False) 
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False) 
 
    input_dim = X_train.shape[1] 
    model = ADVNN(input_dim=input_dim, num_classes=len(np.unique(y))).to(DEVICE) 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
 
    # schedule epsilon per epoch (linear increase) 
    eps_schedule = np.linspace(EPS_START, EPS_MAX, num=EPOCHS) 
 
    global PGD_STEP_SIZE 
    if PGD_STEP_SIZE is None: 
        PGD_STEP_SIZE = (EPS_MAX / max(1, PGD_STEPS))  # simple default 
 
    best_acc = 0.0 
    for epoch in range(1, EPOCHS + 1): 
        eps_curr = float(eps_schedule[epoch - 1]) 
        train(model, optimizer, train_loader, epoch, eps_curr) 
        test_acc = evaluate(model, test_loader) 
        print(f"Validation accuracy (clean): {test_acc:.4f}") 
 
        # save best 
        if test_acc > best_acc: 
            best_acc = test_acc 
            torch.save(model.state_dict(), "advnn_best.pth") 
            print("Saved best model (clean val acc improved).") 
 
    print("Training complete. Best clean val acc:", best_acc) 
 
if __name__ == "__main__": 
    main() 
