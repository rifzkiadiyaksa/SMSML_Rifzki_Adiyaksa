# modelling_advance.py - Untuk Penilaian Advance (4 Poin)

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import dagshub
import time # Untuk menghitung waktu training

# 1. Mengatur MLflow Tracking ke server ONLINE (DagsHub)
# Ganti dengan username dan nama repo Anda
dagshub.init(repo_owner='rifzkiadiyaksa', repo_name='SMSML_Rifzki_Adiyaksa', mlflow=True)

mlflow.set_experiment("submission_advance_final_model")

# Memuat data
train_df = pd.read_csv('lung_cancer_train_preprocessed.csv')
test_df = pd.read_csv('lung_cancer_test_preprocessed.csv')
X_train, y_train = train_df.drop('LUNG_CANCER', axis=1), train_df['LUNG_CANCER']
X_test, y_test = test_df.drop('LUNG_CANCER', axis=1), test_df['LUNG_CANCER']

# Parameter terbaik hasil dari tuning di level 'Skilled'
best_n = 9
best_weights = 'distance'

with mlflow.start_run(run_name="knn_final_model_production"):
    
    # MANUAL LOGGING: Mencatat parameter
    mlflow.log_param("n_neighbors", best_n)
    mlflow.log_param("weights", best_weights)
    
    # Latih model dan ukur waktunya
    start_time = time.time()
    model = KNeighborsClassifier(n_neighbors=best_n, weights=best_weights)
    model.fit(X_train, y_train)
    end_time = time.time()
    
    y_pred = model.predict(X_test)
    
    # MANUAL LOGGING: Metrik standar (seperti autolog)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 2. MANUAL LOGGING: Menambahkan minimal 2 nilai/metrik tambahan
    # Metrik Custom 1: Waktu training (detik)
    training_time = end_time - start_time
    mlflow.log_metric("training_duration_seconds", training_time)

    # Metrik Custom 2: Jumlah fitur yang digunakan
    feature_count = X_train.shape[1]
    mlflow.log_metric("feature_count", feature_count)

    # MANUAL LOGGING: Menyimpan model
    mlflow.sklearn.log_model(model, "final_model")
    
    print("\nModel final selesai dilatih.")
    print(f"Akurasi: {accuracy:.4f}")
    print(f"Waktu Training: {training_time:.4f} detik")
    print(f"Jumlah Fitur: {feature_count}")
    print("✅ Log dan artefak telah berhasil dikirim ke DagsHub.")