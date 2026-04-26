import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    
      # ===== TAMBAHKAN KODE INI UNTUK FIX ERROR =====
    if not os.path.exists("mlruns"):
        os.makedirs("mlruns")
    if not os.path.exists("mlruns/0"):
        os.makedirs("mlruns/0")
    if not os.path.exists("mlruns/0/meta.yaml"):
        with open("mlruns/0/meta.yaml", "w") as f:
            f.write("{}")
    # ===== SAMPAI SINI =====
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    # Path ke dataset yang sudah terpisah
    base_path = "credit_risk_dataset_preprocessed"
    
    X_train_path = f"{base_path}/X_train.csv"
    X_test_path = f"{base_path}/X_test.csv"
    y_train_path = f"{base_path}/y_train.csv"
    y_test_path = f"{base_path}/y_test.csv"
    
    # Cek apakah file ada
    for path in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
    
    # Load dataset
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    print(f"✅ Dataset loaded:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test: {y_test.shape}")
    
    # Handle missing values jika ada
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Print results
        print("\n" + "="*40)
        print(f"🎯 Accuracy : {accuracy:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"🎯 Recall   : {recall:.4f}")
        print(f"🎯 F1 Score : {f1:.4f}")
        print("="*40)
        
        # Save run ID
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)
        
        print(f"\n📝 MLflow Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()