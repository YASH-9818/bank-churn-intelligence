"""
train_models.py
---------------
Run this ONCE to train all models and save them to disk.
Usage: python train_models.py
Outputs: models/  directory with saved .pkl files + models/metrics.json
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data…")
df = pd.read_csv("European_Bank.csv")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
drop_cols = [c for c in ["CustomerId", "Surname", "Year"] if c in df.columns]
df = df.drop(columns=drop_cols)

# One-hot encode
df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=False)

# ── 3. Feature engineering ────────────────────────────────────────────────────
df["BalanceToSalary"]       = df["Balance"] / (df["EstimatedSalary"] + 1)
df["ProductDensity"]        = df["NumOfProducts"] / (df["Tenure"] + 1)
df["EngagementProduct"]     = df["IsActiveMember"] * df["NumOfProducts"]
df["AgeTenureInteraction"]  = df["Age"] * df["Tenure"]
df["HighBalance"]           = (df["Balance"] > 100_000).astype(int)
df["SeniorCustomer"]        = (df["Age"] >= 45).astype(int)

feature_cols = [c for c in df.columns if c != "Exited"]
X = df[feature_cols]
y = df["Exited"]

# Save feature names for app
with open("models/feature_cols.json", "w") as f:
    json.dump(feature_cols, f)

# ── 4. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Models ──────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                        class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=12,
                                        min_samples_leaf=10, class_weight="balanced",
                                        random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                            max_depth=5, subsample=0.8,
                                            random_state=42))
    ]),
}

metrics_all = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in models.items():
    print(f"  Training {name}…")
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv_auc  = cross_val_score(pipe, X_train, y_train, cv=cv,
                               scoring="roc_auc", n_jobs=-1).mean()
    cm      = confusion_matrix(y_test, y_pred).tolist()

    metrics_all[name] = {
        "accuracy":  round(accuracy_score(y_test, y_pred),  4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred),    4),
        "f1":        round(f1_score(y_test, y_pred),        4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba),  4),
        "cv_auc":    round(cv_auc, 4),
        "confusion_matrix": cm,
    }
    print(f"     AUC={metrics_all[name]['roc_auc']:.4f}  "
          f"F1={metrics_all[name]['f1']:.4f}  "
          f"Recall={metrics_all[name]['recall']:.4f}")

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(pipe, f"models/{safe_name}.pkl")

# ── 6. Feature importance from best tree model ────────────────────────────────
rf_clf     = models["Random Forest"].named_steps["clf"]
importance = pd.Series(rf_clf.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False).head(15)
fi_dict    = importance.round(4).to_dict()

with open("models/feature_importance.json", "w") as f:
    json.dump(fi_dict, f, indent=2)

# ── 7. Save all metrics ───────────────────────────────────────────────────────
with open("models/metrics.json", "w") as f:
    json.dump(metrics_all, f, indent=2)

# ── 8. Save test set predictions from best model (Gradient Boosting) ─────────
best_pipe  = models["Gradient Boosting"]
X_test_cp  = X_test.copy()
X_test_cp["Exited"]     = y_test.values
X_test_cp["ChurnProba"] = best_pipe.predict_proba(X_test)[:, 1]
X_test_cp["ChurnPred"]  = best_pipe.predict(X_test)
X_test_cp.to_csv("models/test_predictions.csv", index=False)

print("\n✓ All models trained and saved to models/")
print(json.dumps({k: v["roc_auc"] for k, v in metrics_all.items()}, indent=2))