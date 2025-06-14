# modelling_tuning.py (Versi Perbaikan Final Definitif untuk Google Colab)

import pandas as pd
import mlflow
import mlflow.sklearn
import os
import getpass
import joblib  # Import library joblib untuk menyimpan model
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Langkah 1: Pengaturan Manual Koneksi ke DagsHub ---
# Metode ini adalah yang paling stabil untuk Colab.
os.environ['MLFLOW_TRACKING_USERNAME'] = 'rifzkiadiyaksa'
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass.getpass('Masukkan DagsHub Access Token Anda: ')
mlflow.set_tracking_uri('https://dagshub.com/rifzkiadiyaksa/SMSML_Rifzki_Adiyaksa.mlflow')
print("Koneksi ke DagsHub MLflow Tracking Server telah dikonfigurasi.")

experiment_name = "Lung_Cancer_Prediction_Tuning"
mlflow.set_experiment(experiment_name)
print(f"Eksperimen diatur ke '{experiment_name}'")

# --- Langkah 2: Memuat Data ---
try:
    train_data = pd.read_csv('lung_cancer_train_preprocessed.csv')
    X_train = train_data.drop('LUNG_CANCER', axis=1)
    y_train = train_data['LUNG_CANCER']
    print("Data latih berhasil dimuat.")
except FileNotFoundError:
    print("Error: File 'lung_cancer_train_preprocessed.csv' tidak ditemukan. Mohon upload file tersebut ke Colab.")
    exit()

# --- Langkah 3: Proses Tuning dengan GridSearchCV ---
model = LogisticRegression(max_iter=1000, random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
print("GridSearchCV siap dijalankan.")

# --- BLOK 1: Menjalankan dan Mencatat Hasil Tuning ---
with mlflow.start_run(run_name="GridSearchCV_Parent_Run") as parent_run:
    print(f"\nMemulai Parent Run untuk Tuning: {parent_run.info.run_id}")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV selesai.")

    print("\nMencatat hasil tuning ke DagsHub...")
    mlflow.log_param("best_params", grid_search.best_params_)
    mlflow.log_metric("best_accuracy_cv", grid_search.best_score_)
    
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        with mlflow.start_run(run_name=f"Child_Run_{i}", nested=True) as child_run:
            mlflow.log_params(cv_results['params'][i])
            mlflow.log_metric("mean_test_score", cv_results['mean_test_score'][i])
    
    print("Semua hasil tuning telah dicatat.")
print("Parent run untuk tuning selesai.")

# --- BLOK 2: Mencatat Model Terbaik di Run Terpisah (METODE BARU YANG STABIL) ---
print("\nMemulai run baru untuk menyimpan model terbaik...")
with mlflow.start_run(run_name="Final_Best_Model_Artifact") as final_run:
    print(f"Run untuk model final: {final_run.info.run_id}")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    # Log parameter dan metrik terbaik
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_accuracy)
    print(f"Mencatat ulang parameter terbaik: {best_params}")
    print(f"Mencatat ulang akurasi terbaik: {best_accuracy:.4f}")

    # ===== PERUBAHAN UTAMA DI SINI =====
    # 1. Simpan model ke file lokal terlebih dahulu
    model_filename = "best_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Model terbaik disimpan secara lokal sebagai '{model_filename}'")

    # 2. Log file tersebut sebagai artefak tunggal
    #    Fungsi mlflow.log_artifact() jauh lebih sederhana dan stabil.
    mlflow.log_artifact(model_filename, "model")
    print(f"'{model_filename}' telah berhasil di-log sebagai artefak ke DagsHub di dalam folder 'model'.")
    # ====================================

print("\nSemua proses selesai dengan sukses!")