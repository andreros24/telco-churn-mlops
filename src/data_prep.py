"""
Preprocessing script for Telco Customer Churn dataset.
Import raw data, performing data cleaning and transformations, save:
- the processed dataset (for training)
- fitted preprocessor (to use it in the API stage)
"""

# import libraries and data
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

RAW_PATH = Path('data/raw/telco_churn.csv')
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col='customerID')
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is type object we need to convert to numeric with 0
    # for the missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Target: Yes/No -> 1/0
    df['Churn']=[1 if x=='Yes' else 0 for x in df['Churn']]

    return df

def get_feature_lists(df: pd.DataFrame):
    # the only numerical variable are tenure and Monthly/Total Charges, the other are categorical
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        col for col in df.columns
        if col not in numeric_features + ["Churn"]
    ]
    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    # standardize the numerical variable
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Encode the categorical variable with one-hot encoder
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    return preprocessor

def main():
    print("Loading data...")
    df = load_raw_data(RAW_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    numeric_features, categorical_features = get_feature_lists(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # we split the train and test such that the class weights are the same in the 2 groups
    print("Split train/test ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Preprocessor fitting on training data...")
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Restoring columns name after the encoding
    feature_names = preprocessor.get_feature_names_out()

    train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_df["Churn"] = y_train.values

    test_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_df["Churn"] = y_test.values

    print("Saving processed data...")
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("Saving preprocessor...")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    print(f"Done. Train shape: {train_df.shape}, Test shape: {test_df.shape}")


if __name__ == "__main__":
    main()