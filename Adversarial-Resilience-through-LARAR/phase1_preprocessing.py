import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset from Parquet file
file_path = "/home/kali/Desktop/UNSW_NB15_testing-set.parquet"  # Update path if needed
data = pd.read_parquet(file_path)
print("Initial data preview:")
print(data.head())

# Step 2: Encode categorical features using Label Encoding
categorical_cols = ['proto', 'service', 'state', 'attack_cat']  # based on dataset
for col in categorical_cols:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Step 3: Handle missing and infinite values
# Replace missing values with the column name (as string)
for col in data.columns:
    if data[col].isnull().any():
        data[col].fillna(col, inplace=True)

# Replace infinite values with 0
data.replace([float('inf'), -float('inf')], 0, inplace=True)

# Step 4: Normalize numerical features (z-score normalization)
exclude_columns = ['attack_cat', 'label']
features = data.drop(columns=exclude_columns, errors='ignore')

# Normalize only numeric columns
numeric_cols = features.select_dtypes(include=['number']).columns
scaler = StandardScaler()

features_scaled = features.copy()
for col in numeric_cols:
    features_scaled[col] = scaler.fit_transform(features[[col]])

# Replace original feature columns with normalized values
data.loc[:, features_scaled.columns] = features_scaled

# Step 5: Stratified train-test split (70/30)
X = data.drop(columns=['label'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Test set size:", X_test.shape)
