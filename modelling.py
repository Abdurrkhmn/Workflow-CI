# modelling.py
# ============================================
# MODEL TRAINING FOR CI WORKFLOW
# ============================================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("CI_Experiment")

print("="*60)
print("🚀 CI MODEL TRAINING")
print("="*60)

# Load data
print("\n📂 Loading preprocessed data...")
X_train = pd.read_csv('preprocessing/X_train.csv')
X_test = pd.read_csv('preprocessing/X_test.csv')
y_train = pd.read_csv('preprocessing/y_train.csv').values.ravel()
y_test = pd.read_csv('preprocessing/y_test.csv').values.ravel()

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Train model
print("\n🔧 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

# MLflow logging
with mlflow.start_run(run_name="CI_RandomForest"):
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", None)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Save model locally
    joblib.dump(model, 'model.pkl')
    mlflow.log_artifact('model.pkl')

print("\n✅ Model saved to MLflow!")
print(f"📌 Run ID: {mlflow.active_run().info.run_id}")