"""
data_processing.py

Preprocessing script for the Early-Stage Diabetes Symptoms dataset.

Loads the CSV file
Removes rows with missing values
One-hot encodes all categorical columns
Leaves numeric columns untouched
Converts class labels: "Positive" to 1, "Negative" to 0
Splits the dataset into training and testing sets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

#Constants
TARGET_COLUMN    = "class"
DEFAULT_CSV_PATH = "diabetes_data_upload.csv"
LABEL_MAP        = {"Negative": 0, "Positive": 1}

###########
def load_data(csv_path):                #load data

    return pd.read_csv(csv_path)

def check_missing_values(df):           #Prints missing value counts for each column, if any.
    print("\nChecking for missing values...")
    for col in df.columns:
        count = df[col].isna().sum()
        if count > 0:
            print(f" {col}: {count} missing")

def drop_missing_rows(df): #Drops rows that contain any missing values and prints row counts before and after
    before = len(df)
    df_cleaned = df.dropna()
    after = len(df_cleaned)
    print(f"\nRows before removing missing values: {before}")
    print(f"Rows after  removing missing values: {after}")
    return df_cleaned

def build_preprocessor(df):
    """
    Builds a preprocessing transformer:
    - One-hot encodes all categorical (object-type) columns
    - Keeps numeric columns unchanged
    """

    # Step 1: Find all categorical columns (those with string values)
    categorical_columns = []
    for column in df.columns:
        if df[column].dtype == "object" and column != TARGET_COLUMN:
            categorical_columns.append(column)

    # Step 2: Find all numeric columns (excluding the target)
    numeric_columns = []
    for column in df.columns:
        if column != TARGET_COLUMN and column not in categorical_columns:
            numeric_columns.append(column)

    # Step 3: Print out counts for user clarity
    print("\nDetected categorical columns :", len(categorical_columns))
    print("Detected numeric columns      :", len(numeric_columns))

    # Step 4: Define pipeline for categorical columns
    cat_encoder = OneHotEncoder(handle_unknown="ignore")
    cat_pipeline = Pipeline([
        ("onehot", cat_encoder)
    ])

    # Step 5: Create final ColumnTransformer
    preprocessor = ColumnTransformer([
        ("categorical", cat_pipeline, categorical_columns),
        ("numeric", "passthrough", numeric_columns)
    ])

    return preprocessor


def prepare_data(
    csv_path: str = DEFAULT_CSV_PATH,
    test_size: float = 0.2,
    random_state: int = 70
):
    """
    Full preprocessing routine:
    Load data
    Clean missing values
    Map labels to binary (0/1)
    Split into training and test sets
    Build and return the preprocessor

    Returns:
    - X_train, X_test: Features for train/test
    - y_train, y_test: Labels for train/test
    - preprocessor   : Transformer to use with ML models
    """
    print("\nLoading data...")
    df = load_data(csv_path)

    check_missing_values(df)
    df = drop_missing_rows(df)

    # Separate features and labels
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map(LABEL_MAP)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(df)

    return X_train, X_test, y_train, y_test, preprocessor


# If the script is run directly
if __name__ == "__main__":
    prepare_data(DEFAULT_CSV_PATH)
