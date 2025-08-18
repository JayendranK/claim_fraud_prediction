import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --------------------------
# Configurations
# --------------------------
DATA_DIR = "data/processed"
TARGET_COL = "PotentialFraud"

def load_data():
    """Load train and test datasets and drop rows with null values."""
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    # # Drop rows with any null values
    # train_df.dropna(inplace=True)
    # test_df.dropna(inplace=True)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    print("✅ Data loaded. Null rows dropped.")
    return X_train, X_test, y_train, y_test



def build_preprocessor(X_train):
    """Build preprocessing pipeline with scaling for numeric and one-hot encoding for categorical features."""
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    # Limit to categorical features with < 100 unique values
    categorical_cols = [col for col in categorical_cols if X_train[col].nunique() < 100]

    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])
    return preprocessor
