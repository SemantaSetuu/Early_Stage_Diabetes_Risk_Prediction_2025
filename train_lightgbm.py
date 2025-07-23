"""
train_lightgbm.py
-----------------
• Performs 10-fold Cross Validation on the full dataset (520 rows)
• Performs an 80/20 train-test split evaluation (random_state = 70)
• Saves TWO models:
   – lightgbm_symptom_full.pkl  (trained on 100% data)
   – lightgbm_symptom_80_20.pkl (trained on the 80% split)
"""

import warnings
import joblib

from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report
)

from data_processing import (
    load_data, build_preprocessor, prepare_data,
    TARGET_COLUMN, DEFAULT_CSV_PATH, LABEL_MAP
)

# Disable warnings from LightGBM and sklearn
warnings.filterwarnings("ignore")

# LightGBM model settings
lgbm_params = {
    "objective": "binary",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "random_state": 42,
    "verbosity": -1
}

# Load full dataset and prepare the pipeline
full_df = load_data(DEFAULT_CSV_PATH).dropna()
y_full = full_df[TARGET_COLUMN].map(LABEL_MAP)  # Convert class labels to 0/1
X_full = full_df.drop(columns=[TARGET_COLUMN])
preprocessor = build_preprocessor(full_df)

# Create full pipeline: Preprocessor + LightGBM model
pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LGBMClassifier(**lgbm_params))
])


# 10-Fold Cross Validation (Stratified)
print("Running 10-fold cross-validation…")
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define evaluation metrics
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Perform cross-validation
cv_results = cross_validate(
    pipeline, X_full, y_full,
    cv=cv, scoring=metrics, n_jobs=-1
)

# Print mean ± standard deviation for each metric
for metric in metrics:
    scores = cv_results[f"test_{metric}"]
    print(f"{metric:9s}: {scores.mean():.3f}  (+/- {scores.std():.3f})")

# Generate full classification report from CV predictions
print("\nFull classification report (10-fold CV):")
y_pred_cv = cross_val_predict(pipeline, X_full, y_full, cv=cv)
print(classification_report(y_full, y_pred_cv, digits=3))


# Train model on all data and save
print("\nTraining on entire dataset and saving model...")
pipeline.fit(X_full, y_full)
joblib.dump(pipeline, "lightgbm_symptom_full.pkl")
print("Saved to lightgbm_symptom_full.pkl")


# 80/20 Hold-out Evaluation (random_state = 70)
print("\nRunning 80 / 20 hold-out evaluation…")
X_train, X_test, y_train, y_test, preprocessor_hold = prepare_data(
    csv_path=DEFAULT_CSV_PATH,
    test_size=0.2,
    random_state=40
)

# New pipeline for hold-out
holdout_model = LGBMClassifier(**{**lgbm_params, "random_state": 40})
holdout_pipeline = Pipeline([
    ("prep", preprocessor_hold),
    ("clf", holdout_model)
])

# Train and evaluate
holdout_pipeline.fit(X_train, y_train)
y_pred = holdout_pipeline.predict(X_test)
y_proba = holdout_pipeline.predict_proba(X_test)[:, 1]

print("\nClassification report (80 / 20):")
print(classification_report(y_test, y_pred, digits=3))
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.3f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Save 80/20 model
joblib.dump(holdout_pipeline, "lightgbm_symptom_80_20.pkl")
print("Saved hold-out model to lightgbm_symptom_80_20.pkl")
