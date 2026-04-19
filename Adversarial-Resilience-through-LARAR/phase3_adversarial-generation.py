import os 
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, 
confusion_matrix 
 
# --------------------------- 
# Config / Hyperparameters 
# --------------------------- 
DATA_PATH = "/home/kali/Desktop/UNSW_NB15_testing-set.parquet" 
MODEL_PATH = "advnn_best.pth"  # Path to trained model from Phase 2 
BATCH_SIZE = 256 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
# Attack parameters 
PGD_EPSILON = 0.3        # L-inf perturbation bound 
PGD_STEPS = 40           # Number of PGD iterations 
PGD_STEP_SIZE = 0.01     # Step size for each iteration 
FGSM_EPSILON = 0.3       # FGSM perturbation bound 
 
# Transfer attack parameters 
TRANSFER_EPSILON = 0.3 
TRANSFER_STEPS = 40 
TRANSFER_STEP_SIZE = 0.01 
 
SEED = 42 
torch.manual_seed(SEED) 
np.random.seed(SEED) 
 
# --------------------------- 
# Model Architecture (same as Phase 2) 
# --------------------------- 
class ADVNN(nn.Module): 
    def __init__(self, input_dim: int, h1: int = 256, h2: int = 128, num_classes: 
int = 2): 
        super().__init__() 
        self.fc1 = nn.Linear(input_dim, h1) 
        self.bn1 = nn.BatchNorm1d(h1) 
        self.fc2 = nn.Linear(h1, h2) 
        self.bn2 = nn.BatchNorm1d(h2) 
        self.fc_out = nn.Linear(h2, num_classes) 
 
    def forward(self, x, return_features: bool = False): 
        x = self.fc1(x) 
        x = self.bn1(x) 
        x = F.relu(x) 
        feat1 = x 
        x = self.fc2(x) 
        x = self.bn2(x) 
        x = F.relu(x) 
        feat2 = x 
        out = self.fc_out(x) 
        if return_features: 
            return out, feat1, feat2  
        return out 
 
# Surrogate model for transfer attacks (different architecture) 
class SurrogateModel(nn.Module): 
    def __init__(self, input_dim: int, num_classes: int = 2): 
        super().__init__() 
        self.fc1 = nn.Linear(input_dim, 128) 
        self.fc2 = nn.Linear(128, 64) 
        self.fc3 = nn.Linear(64, num_classes) 
         
    def forward(self, x): 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        return self.fc3(x) 
 
# --------------------------- 
# Data Loading (same as Phase 1) 
# --------------------------- 
def load_and_preprocess(path: str): 
    """Load and preprocess UNSW-NB15 dataset""" 
    if path.endswith(".parquet"): 
        data = pd.read_parquet(path) 
    else: 
        data = pd.read_csv(path) 
     
    # Encode categorical features 
    categorical_cols = ['proto', 'service', 'state', 'attack_cat'] 
    for col in categorical_cols: 
        if col in data.columns: 
            le = LabelEncoder() 
            data[col] = le.fit_transform(data[col].astype(str)) 
     
    # Handle missing values 
    for col in data.columns: 
        if data[col].isnull().any(): 
            data[col].fillna(col, inplace=True) 
     
    # Replace infinite values 
    data.replace([float('inf'), -float('inf')], 0, inplace=True) 
     
    # Separate features and labels 
    exclude_columns = ['attack_cat', 'label'] 
    features = data.drop(columns=exclude_columns, errors='ignore') 
     
    # Normalize numerical features 
    numeric_cols = features.select_dtypes(include=['number']).columns 
    scaler = StandardScaler() 
    features_scaled = features.copy() 
 
    features_scaled[numeric_cols] = scaler.fit_transform(features[numeric_cols]) 
     
    X = features_scaled.to_numpy(dtype=np.float32) 
    y = data['label'].to_numpy(dtype=np.int64) 
     
    return X, y 
 
# --------------------------- 
# Adversarial Attack Methods 
# --------------------------- 
 
def fgsm_attack(model, x, y, epsilon, device): 
    """ 
    Fast Gradient Sign Method (FGSM) 
    Single-step attack using sign of gradient 
     
    Args: 
        model: Target model 
        x: Input samples (tensor) 
        y: True labels (tensor) 
        epsilon: Perturbation bound (L-inf) 
        device: torch device 
     
    Returns: 
        Adversarial examples 
    """ 
    model.eval() 
    x_adv = x.clone().detach().requires_grad_(True) 
     
    # Forward pass 
    outputs = model(x_adv) 
    loss = F.cross_entropy(outputs, y) 
     
    # Backward pass to get gradients 
    loss.backward() 
     
    # Generate adversarial example using sign of gradient 
    grad_sign = x_adv.grad.sign() 
    x_adv = x_adv.detach() + epsilon * grad_sign 
     
    return x_adv.detach() 
 
def pgd_attack(model, x, y, epsilon, steps, step_size, device, random_start=True): 
    """ 
    Projected Gradient Descent (PGD) Attack 
    Multi-step iterative attack with projection 
     
    Args: 
 
        model: Target model 
        x: Input samples (tensor) 
        y: True labels (tensor) 
        epsilon: Perturbation bound (L-inf) 
        steps: Number of iterations 
        step_size: Step size per iteration 
        device: torch device 
        random_start: Whether to start from random point in epsilon ball 
     
    Returns: 
        Adversarial examples 
    """ 
    model.eval() 
    x_orig = x.clone().detach() 
     
    # Random initialization within epsilon ball 
    if random_start: 
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon) 
        x_adv = torch.clamp(x_adv, 0, 1)  # Ensure valid range if needed 
    else: 
        x_adv = x_orig.clone() 
     
    # Iterative attack 
    for i in range(steps): 
        x_adv = x_adv.clone().detach().requires_grad_(True) 
         
        # Forward pass 
        outputs = model(x_adv) 
        loss = F.cross_entropy(outputs, y) 
         
        # Backward pass 
        loss.backward() 
         
        # Update adversarial example 
        grad = x_adv.grad.detach() 
        x_adv = x_adv.detach() + step_size * torch.sign(grad) 
         
        # Project back to epsilon ball around original input 
        perturbation = torch.clamp(x_adv - x_orig, -epsilon, epsilon) 
        x_adv = x_orig + perturbation 
     
    return x_adv.detach() 
 
def transfer_attack(surrogate_model, target_model, x, y, epsilon, steps, step_size, 
device): 
    """ 
    Transfer Attack 
    Generate adversarial examples on surrogate model and test on target model 
 
     
    Args: 
        surrogate_model: Model used to generate adversarial examples 
        target_model: Target model to attack 
        x: Input samples (tensor) 
        y: True labels (tensor) 
        epsilon: Perturbation bound 
        steps: Number of PGD steps 
        step_size: Step size per iteration 
        device: torch device 
     
    Returns: 
        Adversarial examples, source accuracy, transfer accuracy 
    """ 
    # Generate adversarial examples using surrogate model 
    x_adv = pgd_attack(surrogate_model, x, y, epsilon, steps, step_size, device) 
     
    # Evaluate on surrogate model 
    surrogate_model.eval() 
    with torch.no_grad(): 
        surrogate_outputs = surrogate_model(x_adv) 
        surrogate_preds = surrogate_outputs.argmax(dim=1) 
        source_acc = (surrogate_preds == y).float().mean().item() 
     
    # Evaluate on target model 
    target_model.eval() 
    with torch.no_grad(): 
        target_outputs = target_model(x_adv) 
        target_preds = target_outputs.argmax(dim=1) 
        transfer_acc = (target_preds == y).float().mean().item() 
     
    return x_adv, source_acc, transfer_acc 
 
# --------------------------- 
# Evaluation Functions 
# --------------------------- 
 
def evaluate_model(model, loader, device, attack_type=None, attack_params=None): 
    """ 
    Evaluate model on clean or adversarial samples 
     
    Args: 
        model: Model to evaluate 
        loader: DataLoader 
        device: torch device 
        attack_type: 'clean', 'fgsm', 'pgd', or None 
        attack_params: Dictionary of attack parameters 
     
 
    Returns: 
        Dictionary with metrics 
    """ 
    model.eval() 
    all_preds = [] 
    all_labels = [] 
    total_loss = 0.0 
     
    for x_batch, y_batch in loader: 
        x_batch = x_batch.to(device) 
        y_batch = y_batch.to(device) 
         
        # Generate adversarial examples if specified 
        if attack_type == 'fgsm': 
            x_batch = fgsm_attack(model, x_batch, y_batch,  
                                 attack_params['epsilon'], device) 
        elif attack_type == 'pgd': 
            x_batch = pgd_attack(model, x_batch, y_batch, 
                                attack_params['epsilon'], 
                                attack_params['steps'], 
                                attack_params['step_size'], 
                                device) 
         
        # Evaluate 
        with torch.no_grad(): 
            outputs = model(x_batch) 
            loss = F.cross_entropy(outputs, y_batch) 
            total_loss += loss.item() * x_batch.size(0) 
            preds = outputs.argmax(dim=1) 
             
            all_preds.extend(preds.cpu().numpy()) 
            all_labels.extend(y_batch.cpu().numpy()) 
     
    # Calculate metrics 
    all_preds = np.array(all_preds) 
    all_labels = np.array(all_labels) 
     
    accuracy = accuracy_score(all_labels, all_preds) 
    precision = precision_score(all_labels, all_preds, average='weighted', 
zero_division=0) 
    recall = recall_score(all_labels, all_preds, average='weighted', 
zero_division=0) 
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) 
    avg_loss = total_loss / len(all_labels) 
     
    return { 
        'accuracy': accuracy, 
        'precision': precision, 
 
        'recall': recall, 
        'f1_score': f1, 
        'loss': avg_loss, 
        'predictions': all_preds, 
        'labels': all_labels 
    } 
 
def generate_adversarial_dataset(model, loader, device, attack_type, attack_params): 
    """ 
    Generate complete adversarial dataset 
     
    Returns: 
        X_adv (numpy array), y (numpy array) 
    """ 
    model.eval() 
    all_x_adv = [] 
    all_y = [] 
     
    for x_batch, y_batch in loader: 
        x_batch = x_batch.to(device) 
        y_batch = y_batch.to(device) 
         
        if attack_type == 'fgsm': 
            x_adv = fgsm_attack(model, x_batch, y_batch,  
                               attack_params['epsilon'], device) 
        elif attack_type == 'pgd': 
            x_adv = pgd_attack(model, x_batch, y_batch, 
                              attack_params['epsilon'], 
                              attack_params['steps'], 
                              attack_params['step_size'], 
                              device) 
        else: 
            x_adv = x_batch 
         
        all_x_adv.append(x_adv.cpu().numpy()) 
        all_y.append(y_batch.cpu().numpy()) 
     
    X_adv = np.vstack(all_x_adv) 
    y = np.concatenate(all_y) 
     
    return X_adv, y 
 
# --------------------------- 
# Main Execution 
# --------------------------- 
 
def main(): 
    print("=" * 80) 
 
    print("Phase 3: Adversarial Sample Generation and Evaluation") 
    print("=" * 80) 
     
    # Load data 
    print("\n[1] Loading and preprocessing data...") 
    X, y = load_and_preprocess(DATA_PATH) 
    print(f"    Dataset shape: X={X.shape}, y={y.shape}") 
     
    # Train/test split (same as Phase 1 & 2) 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.3, stratify=y, random_state=SEED 
    ) 
     
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)) 
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) 
     
    # Load trained model from Phase 2 
    print("\n[2] Loading trained model from Phase 2...") 
    input_dim = X_train.shape[1] 
    num_classes = len(np.unique(y)) 
    model = ADVNN(input_dim=input_dim, num_classes=num_classes).to(DEVICE) 
     
    if os.path.exists(MODEL_PATH): 
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)) 
        print(f"    Model loaded from {MODEL_PATH}") 
    else: 
        print(f"    WARNING: Model file {MODEL_PATH} not found!") 
        print(f"    Please train the model using Phase 2 first.") 
        return 
     
    # Evaluate on clean data 
    print("\n[3] Evaluating on CLEAN test data...") 
    clean_results = evaluate_model(model, test_loader, DEVICE) 
    print(f"    Clean Accuracy:  {clean_results['accuracy']:.4f}") 
    print(f"    Clean Precision: {clean_results['precision']:.4f}") 
    print(f"    Clean Recall:    {clean_results['recall']:.4f}") 
    print(f"    Clean F1-Score:  {clean_results['f1_score']:.4f}") 
     
    # FGSM Attack Evaluation 
    print("\n[4] Evaluating FGSM Attack...") 
    fgsm_params = {'epsilon': FGSM_EPSILON} 
    fgsm_results = evaluate_model(model, test_loader, DEVICE,  
                                  attack_type='fgsm', attack_params=fgsm_params) 
    print(f"    FGSM Accuracy:  {fgsm_results['accuracy']:.4f}") 
    print(f"    FGSM Precision: {fgsm_results['precision']:.4f}") 
    print(f"    FGSM Recall:    {fgsm_results['recall']:.4f}") 
    print(f"    FGSM F1-Score:  {fgsm_results['f1_score']:.4f}") 
    print(f"    Attack Success Rate: {1 - fgsm_results['accuracy']:.4f}") 
 
     
    # PGD Attack Evaluation 
    print("\n[5] Evaluating PGD Attack...") 
    pgd_params = { 
        'epsilon': PGD_EPSILON, 
        'steps': PGD_STEPS, 
        'step_size': PGD_STEP_SIZE 
    } 
    pgd_results = evaluate_model(model, test_loader, DEVICE, 
                                 attack_type='pgd', attack_params=pgd_params) 
    print(f"    PGD Accuracy:  {pgd_results['accuracy']:.4f}") 
    print(f"    PGD Precision: {pgd_results['precision']:.4f}") 
    print(f"    PGD Recall:    {pgd_results['recall']:.4f}") 
    print(f"    PGD F1-Score:  {pgd_results['f1_score']:.4f}") 
    print(f"    Attack Success Rate: {1 - pgd_results['accuracy']:.4f}") 
     
    # Transfer Attack Evaluation 
    print("\n[6] Evaluating Transfer Attack...") 
    print("    Training surrogate model...") 
     
    # Train a simple surrogate model 
    train_dataset = TensorDataset(torch.from_numpy(X_train), 
torch.from_numpy(y_train)) 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
     
    surrogate = SurrogateModel(input_dim=input_dim, 
num_classes=num_classes).to(DEVICE) 
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=0.001) 
     
# Quick training of surrogate (5 epochs)
surrogate.train()

for epoch in range(5):
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = surrogate(x_batch)

        loss = F.cross_entropy(outputs, y_batch)
        loss.backward()
        optimizer.step()

print("    Generating transfer attacks...")

transfer_params = {
    'epsilon': TRANSFER_EPSILON,
    'steps': TRANSFER_STEPS,
    'step_size': TRANSFER_STEP_SIZE
}

all_transfer_source_acc = []
all_transfer_target_acc = []

for x_batch, y_batch in test_loader:
    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

    _, src_acc, tgt_acc = transfer_attak(
        surrogate, model, x_batch, y_batch,
        transfer_params['epsilon'],
        transfer_params['steps'],
        transfer_params['step_size'],
        DEVICE
    )

    all_transfer_source_acc.append(src_acc)
    all_transfer_target_acc.append(tgt_acc)

avg_sorce_acc = np.mean(all_transfer_source_acc)
avg_transfer_acc = np.mean(all_transfer_target_acc)

print(f"    Surrogate Model Accuracy (on adv): {avg_source_acc:.4f}")
print(f"    Target Model Accuracy (transfer):  {avg_transfer_acc:.4f}")
print(f"    Transfer Success Rate: {1 - avg_transfer_acc:.4f}")
     
    # Generate and save adversarial datasets 
    print("\n[7] Generating adversarial datasets...") 
     
    print("    Generating FGSM adversarial samples...") 
    X_fgsm, y_fgsm = generate_adversarial_dataset( 
        model, test_loader, DEVICE, 'fgsm', fgsm_params 
    ) 
    np.save('adversarial_fgsm_X.npy', X_fgsm) 
    np.save('adversarial_fgsm_y.npy', y_fgsm) 
    print(f"    Saved FGSM dataset: {X_fgsm.shape}") 
     
    print("    Generating PGD adversarial samples...") 
    X_pgd, y_pgd = generate_adversarial_dataset( 
        model, test_loader, DEVICE, 'pgd', pgd_params 
    ) 
    np.save('adversarial_pgd_X.npy', X_pgd) 
    np.save('adversarial_pgd_y.npy', y_pgd) 
    print(f"    Saved PGD dataset: {X_pgd.shape}") 
     
    # Summary Report 
    print("\n" + "=" * 80) 
    print("ROBUSTNESS EVALUATION SUMMARY") 
    print("=" * 80) 
    print(f"{'Attack Type':<20} {'Accuracy':<12} {'F1-Score':<12} {'Success 
Rate':<12}") 
    print("-" * 80)  
    print(f"{'Clean':<20} {clean_results['accuracy']:<12.4f} 
{clean_results['f1_score']:<12.4f} {'N/A':<12}") 
    print(f"{'FGSM (ε=0.3)':<20} {fgsm_results['accuracy']:<12.4f} 
{fgsm_results['f1_score']:<12.4f} {1-fgsm_results['accuracy']:<12.4f}") 
    print(f"{'PGD (ε=0.3)':<20} {pgd_results['accuracy']:<12.4f} 
{pgd_results['f1_score']:<12.4f} {1-pgd_results['accuracy']:<12.4f}") 
    print(f"{'Transfer Attack':<20} {avg_transfer_acc:<12.4f} {'N/A':<12} {1
avg_transfer_acc:<12.4f}") 
    print("=" * 80) 
     
 
if __name__ == "__main__": 
    main()
