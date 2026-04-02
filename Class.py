import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import pickle
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load & Clean Data ──────────────────────────────────────────────
df = pd.read_csv('Gender.csv')
df = df.dropna(axis=1, how='all')
df = df.dropna()
print("Dataset shape:", df.shape)
print(df.to_string(index=False))

# ── 2. Features & Target ──────────────────────────────────────────────
X = df[['skirt', 'hair', 'frequency']]
y = df['gender']

print(f"\nClass distribution:\n{y.value_counts().rename({0: 'Boy', 1: 'Girl'})}")

# ── 3. XGBoost Model ──────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)

# ── 4. Leave-One-Out Cross Validation ────────────────────────────────
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

# Get LOO predictions for confusion matrix & full metrics
y_pred_loo = cross_val_predict(model, X, y, cv=loo)

print(f"\n{'='*45}")
print(f"         MODEL EVALUATION METRICS")
print(f"{'='*45}")
print(f"  Method        : Leave-One-Out Cross Validation")
print(f"  Accuracy      : {accuracy_score(y, y_pred_loo) * 100:.1f}%")
print(f"  Precision     : {precision_score(y, y_pred_loo):.3f}")
print(f"  Recall        : {recall_score(y, y_pred_loo):.3f}")
print(f"  F1 Score      : {f1_score(y, y_pred_loo):.3f}")
print(f"  Std Deviation : {scores.std() * 100:.1f}%")
print(f"{'='*45}")

print(f"\n── Classification Report ──")
print(classification_report(y, y_pred_loo, target_names=['Boy', 'Girl']))

print(f"── Confusion Matrix ──")
cm = confusion_matrix(y, y_pred_loo)
print(f"               Predicted")
print(f"               Boy   Girl")
print(f"Actual  Boy  [  {cm[0][0]}      {cm[0][1]}  ]")
print(f"        Girl [  {cm[1][0]}      {cm[1][1]}  ]")

# ── 5. Train Final Model on ALL Data ─────────────────────────────────
model.fit(X, y)

print(f"\n── Feature Importance ──")
for feat, imp in zip(X.columns, model.feature_importances_):
    bar = '█' * int(imp * 30)
    print(f"{feat:<12} {bar} {imp:.3f}")

# ── 6. Save Model ─────────────────────────────────────────────────────
model_filename = 'gender_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)
print(f"\n✅ Model saved as '{model_filename}'")

# ── 7. Load & Predict with Saved Model ───────────────────────────────
with open(model_filename, 'rb') as f:
    loaded_model = pickle.load(f)
print(f"✅ Model loaded successfully!")

print(f"\n── Predict New Samples ──")
new_people = pd.DataFrame({
    'skirt':     [1, 0, 0],
    'hair':      [1, 0, 1],
    'frequency': [250, 110, 300]
})
preds = loaded_model.predict(new_people)
probs = loaded_model.predict_proba(new_people)
labels = {0: 'Boy 👦', 1: 'Girl 👧'}
print(f"{'Person':<10} {'Skirt':<8} {'Hair':<8} {'Freq(Hz)':<12} {'Prediction':<12} {'Confidence'}")
print("-" * 65)
for i, (_, row) in enumerate(new_people.iterrows()):
    conf = max(probs[i]) * 100
    print(f"Person {i+1:<4} {int(row.skirt):<8} {int(row.hair):<8} "
          f"{int(row.frequency):<12} {labels[preds[i]]:<12} {conf:.1f}%")