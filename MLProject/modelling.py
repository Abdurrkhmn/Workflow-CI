# modelling.py untuk Workflow-CI
import mlflow
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dagshub

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

# Setup DagsHub tracking (optional, untuk advance)
dagshub.init(repo_owner='Abdurrkhmn', repo_name='Eksperimen_SML_Abdurrahman', mlflow=True)

# Load data
X_train = pd.read_csv('credit_risk_dataset_preprocessed/X_train.csv')
X_test = pd.read_csv('credit_risk_dataset_preprocessed/X_test.csv')
y_train = pd.read_csv('credit_risk_dataset_preprocessed/y_train.csv').values.ravel()
y_test = pd.read_csv('credit_risk_dataset_preprocessed/y_test.csv').values.ravel()

with mlflow.start_run(run_name="CI_Workflow_Run") as run:
    mlflow.log_param("n_estimators", args.n_estimators)
    
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
    
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Run ID: {run.info.run_id}")
    print("✅ Model training completed in CI!")