# modelling.py untuk Workflow-CI (Advance ready)
import mlflow
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Gunakan tracking lokal (akan di-capture untuk Docker nanti)
mlflow.set_tracking_uri("file:./mlruns")

# Load data
base_path = os.path.dirname(os.path.abspath(__file__))
X_train = pd.read_csv(os.path.join(base_path, 'credit_risk_dataset_preprocessed/X_train.csv'))
X_test = pd.read_csv(os.path.join(base_path, 'credit_risk_dataset_preprocessed/X_test.csv'))
y_train = pd.read_csv(os.path.join(base_path, 'credit_risk_dataset_preprocessed/y_train.csv')).values.ravel()
y_test = pd.read_csv(os.path.join(base_path, 'credit_risk_dataset_preprocessed/y_test.csv')).values.ravel()

print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")

with mlflow.start_run(run_name="CI_Workflow_Run") as run:
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("model_type", "RandomForestClassifier")
    
    # Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Confusion Matrix (artifact tambahan)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ SUCCESS! Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")