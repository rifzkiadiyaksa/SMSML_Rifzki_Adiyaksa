# modelling.py - Untuk Penilaian Basic (2 Poin)

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import os

# Pastikan direktori 'mlruns' ada untuk tracking lokal
if not os.path.exists("mlruns"):
    os.makedirs("mlruns")

# 1. Mengatur MLflow Tracking ke server LOKAL
# Perintah ini akan membuat folder 'mlruns' di direktori Anda
mlflow.set_tracking_uri("file:./mlruns")

# 2. Menentukan nama eksperimen
mlflow.set_experiment("submission_basic_knn")

# 3. Memuat data yang sudah diproses
try:
    train_df = pd.read_csv('lung_cancer_train_preprocessed.csv')
    test_df = pd.read_csv('lung_cancer_test_preprocessed.csv')
except FileNotFoundError:
    print("Pastikan file 'lung_cancer_train_preprocessed.csv' dan 'lung_cancer_test_preprocessed.csv' ada di folder yang sama.")
    exit()

# Pisahkan fitur dan target
X_train = train_df.drop('LUNG_CANCER', axis=1)
y_train = train_df['LUNG_CANCER']
X_test = test_df.drop('LUNG_CANCER', axis=1)
y_test = test_df['LUNG_CANCER']

# 4. Mengaktifkan AUTOLOG dari MLflow
# Ini adalah perintah kunci untuk poin Basic.
# MLflow akan otomatis mencatat parameter, metrik, dan model.
mlflow.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name="knn_basic_model")

with mlflow.start_run(run_name="knn_basic_run"):
    
    # Inisialisasi dan latih model
    # Parameter didefinisikan di sini dan akan otomatis dicatat oleh autolog.
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
    model.fit(X_train, y_train)
    
    # Evaluasi model (skor akurasi juga akan otomatis dicatat)
    accuracy = model.score(X_test, y_test)
    
    print(f"\nModel KNN selesai dilatih.")
    print(f"Akurasi model: {accuracy:.4f}")
    print("✅ Autologging selesai. Periksa MLflow UI untuk melihat hasilnya.")