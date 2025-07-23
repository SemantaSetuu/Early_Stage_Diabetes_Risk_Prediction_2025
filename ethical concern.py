# rf_ethics_checks.py
# ---------------------------------------------------------------
# Ethical‑concern analyses for Early‑Stage‑Diabetes Random‑Forest
# ---------------------------------------------------------------
# • A. Fairness / subgroup‑bias audit (Gender)
# • B. False‑positive / false‑negative cost estimation
# • C. Explainability: permutation importance for deployment model
# ---------------------------------------------------------------

import joblib
import pandas as pd
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Helper paths
# ---------------------------------------------------------------
HOLDOUT_MODEL = "lightgbm_symptom_80_20.pkl"    # trained on 80 % split
DEPLOY_MODEL  = "lightgbm_symptom_full.pkl"     # trained on full data
CSV_PATH      = "diabetes_data_upload.csv"

# ---------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------
if not Path(HOLDOUT_MODEL).exists():
    raise FileNotFoundError(f"Cannot find {HOLDOUT_MODEL}")
if not Path(DEPLOY_MODEL).exists():
    raise FileNotFoundError(f"Cannot find {DEPLOY_MODEL}")

pipe_hold = joblib.load(HOLDOUT_MODEL)
pipe_full = joblib.load(DEPLOY_MODEL)

# ---------------------------------------------------------------
# Load & clean the raw CSV exactly as in data_processing.py
# ---------------------------------------------------------------
from data_processing import (
    load_data, drop_missing_rows, TARGET_COLUMN, LABEL_MAP
)

df_full = load_data(CSV_PATH)
df_full = drop_missing_rows(df_full)

# Recreate the **same** 80 / 20 split used when the hold‑out model
# was trained (random_state = 70)
from sklearn.model_selection import train_test_split
X = df_full.drop(columns=[TARGET_COLUMN])
y = df_full[TARGET_COLUMN].map(LABEL_MAP)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=70
)

# ----------------------------------------------------------------
# A. FAIRNESS AUDIT (Gender)
# ----------------------------------------------------------------
print("\nFairness audit on hold‑out (gender)")
subgroup_col = "Gender"           # simple binary subgroup
results = defaultdict(dict)

# Obtain predictions once
y_pred_te = pipe_hold.predict(X_te)

for grp in ["Male", "Female"]:
    idx = X_te[subgroup_col] == grp
    if idx.sum() == 0:
        continue                 # skip if group absent

    acc  = accuracy_score(y_te[idx], y_pred_te[idx])
    prec = precision_score(y_te[idx], y_pred_te[idx])
    rec  = recall_score(y_te[idx], y_pred_te[idx])
    cm   = confusion_matrix(y_te[idx], y_pred_te[idx])
    fp   = cm[0, 1]
    tn   = cm[0, 0]
    fpr  = fp / (fp + tn)

    results[grp] = {
        "n"        : int(idx.sum()),
        "accuracy" : acc,
        "precision": prec,
        "recall"   : rec,
        "FPR"      : fpr
    }

df_fair = pd.DataFrame(results).T.round(3)
print(df_fair, "\n")

acc_gap  = abs(df_fair.loc["Male", "accuracy"]  - df_fair.loc["Female", "accuracy"])
fpr_gap  = abs(df_fair.loc["Male", "FPR"]       - df_fair.loc["Female", "FPR"])
print(f"Accuracy gap Male‑Female : {acc_gap:.3f}")
print(f"FPR gap Male‑Female      : {fpr_gap:.3f}")

# ----------------------------------------------------------------
# B. COST OF ERRORS
# ----------------------------------------------------------------
print("\n=== B. False‑positive / false‑negative cost ===")
cm_hold = confusion_matrix(y_te, y_pred_te, labels=[0, 1])
tn, fp, fn, tp = cm_hold.ravel()

COST_FP = 15   # €‑cost of unnecessary HbA1c / clinical consult
COST_FN = 200  # €‑cost of missed early diagnosis (conservative)

total_cost = fp * COST_FP + fn * COST_FN
print(f"Confusion Matrix [TN FP; FN TP] =\n{cm_hold}")
print(f"False Positives : {fp}  (cost €{fp*COST_FP})")
print(f"False Negatives : {fn}  (cost €{fn*COST_FN})")
print(f"=> Estimated total cost on 20 % sample: €{total_cost}\n")

# ----------------------------------------------------------------
# C. EXPLAINABILITY (Permutation importance on deployment model)
# ----------------------------------------------------------------
print("=== C. Permutation Importance (deployment model) ===")
# Use the *same* 20 % set for interpretability demo – fine, we’re not tuning
r = permutation_importance(
    pipe_full, X_te, y_te,
    n_repeats=15, random_state=42, n_jobs=-1
)

importances = pd.Series(r.importances_mean, index=pipe_full[:-1].get_feature_names_out())
imp_sorted = importances.sort_values(ascending=False)[:15]

print("\nTop‑15 features by permutation importance:")
print(imp_sorted.round(4))

# Optional – plot
plt.figure(figsize=(8,5))
imp_sorted[::-1].plot(kind="barh")
plt.title("Permutation importance (Random Forest full model)")
plt.xlabel("Mean decrease in accuracy")
plt.tight_layout()
plt.show()
