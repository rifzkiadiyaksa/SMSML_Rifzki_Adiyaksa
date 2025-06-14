# modelling_tuning_optuna.py - VERSI FINAL (DIJAMIN BERHASIL)

import json
import time
import os
import joblib  # <-- TAMBAHKAN IMPORT INI

import dagshub
import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier

# --- PENGATURAN KONEKSI KE DAGSHUB ---
# Pastikan repo_owner dan repo_name sudah sesuai dengan milik Anda
dagshub.init(
    repo_owner="rifzkiadiyaksa", repo_name="SMSML_Rifzki_Adiyaksa", mlflow=True
)
mlflow.set_experiment("KNN Tuning with Optuna (Final Fix)")

# --- MEMUAT DATA ---
try:
    train_df = pd.read_csv('/content/lung_cancer_train_preprocessed.csv')
    test_df = pd.read_csv('/content/lung_cancer_test_preprocessed.csv')
except FileNotFoundError:
    print("Pastikan file CSV hasil preprocessing ('lung_cancer_train_preprocessed.csv' dan 'lung_cancer_test_preprocessed.csv') ada di folder yang sama.")
    exit()

X_train = train_df.drop('LUNG_CANCER', axis=1)
y_train = train_df['LUNG_CANCER']
X_test = test_df.drop('LUNG_CANCER', axis=1)
y_test = test_df['LUNG_CANCER']


# --- FUNGSI OBJECTIVE UNTUK OPTUNA ---
def objective(trial):
    """
    Fungsi ini mendefinisikan satu kali proses training dan evaluasi model
    yang akan dioptimalkan oleh Optuna.
    """
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 21, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical(
            "algorithm", ["ball_tree", "kd_tree", "brute"]
        ),
        "p": trial.suggest_int("p", 1, 2),
    }

    with mlflow.start_run(run_name=f"KNN_Trial_{trial.number}"):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        model = KNeighborsClassifier(**params)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        mlflow.log_metric("training_duration_seconds", train_time)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        model_filename = "model.pkl"
        joblib.dump(model, model_filename)

        mlflow.log_artifact(local_path=model_filename, artifact_path="model")

        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - Trial {trial.number}")
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        mlflow.log_text(json.dumps(metrics_dict, indent=2), "metrics.json")
        
        print(f"Trial {trial.number} selesai. Akurasi: {accuracy:.4f}")

        return accuracy


# --- MENJALANKAN PROSES OPTIMISASI ---
study = optuna.create_study(direction="maximize")
N_TRIALS = 20 
study.optimize(objective, n_trials=N_TRIALS)

print("\n======================================")
print("Proses Tuning Selesai!")
print(f"Jumlah trial: {len(study.trials)}")
print(f"Trial terbaik: Trial ke-{study.best_trial.number}")
print("Parameter terbaik:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
print(f"Akurasi terbaik: {study.best_value:.4f}")
print("======================================")