# modelling_dagshub.py

# Import library yang dibutuhkan
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime
import mlflow
import dagshub
import joblib

# 1. Inisialisasi koneksi ke DagsHub
# Baris ini secara otomatis mengatur MLflow Tracking URI untuk Anda.
dagshub.init(repo_owner='rifzkiadiyaksa', repo_name='SMSML_Rifzki_Adiyaksa', mlflow=True)

# 2. Set nama eksperimen di DagsHub
# Anda bisa memilih nama yang lebih deskriptif untuk eksperimen ini
mlflow.set_experiment("KNN_Production_Model_DagsHub")

# 3. Muat data yang sudah diproses dari Kriteria 1
# Pastikan file CSV berada di direktori yang sama dengan skrip ini
try:
    train_df = pd.read_csv('lung_cancer_train_preprocessed.csv')
    test_df = pd.read_csv('lung_cancer_test_preprocessed.csv')
except FileNotFoundError:
    print("Error: Pastikan file 'lung_cancer_train_preprocessed.csv' dan 'lung_cancer_test_preprocessed.csv' ada di folder yang sama.")
    exit()

# Pisahkan fitur dan target
X_train = train_df.drop('LUNG_CANCER', axis=1)
y_train = train_df['LUNG_CANCER']
X_test = test_df.drop('LUNG_CANCER', axis=1)
y_test = test_df['LUNG_CANCER']

# Contoh input untuk logging model (best practice)
input_example = X_train.head(5)

# 4. Mulai MLflow run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"KNN_DagsHub_Run_{timestamp}"

print(f"Memulai run: {run_name}")

with mlflow.start_run(run_name=run_name):
    # Log parameter yang digunakan untuk run ini
    n_neighbors = 7  # Anda bisa mengubah parameter ini
    algorithm = 'auto'
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("algorithm", algorithm)

    # Latih model machine learning
    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    model.fit(X_train, y_train)

    # Evaluasi model dan log metriknya
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model sebagai artefak di DagsHub
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model", # Ini akan membuat folder 'model' di DagsHub
        input_example=input_example
    )
    
    # (Opsional) Simpan model secara lokal juga jika diperlukan
    joblib.dump(model, "knn_model.pkl")

    print(f"\nRun '{run_name}' selesai.")
    print(f"Akurasi model: {accuracy:.4f}")
    print("✅ Log dan artefak telah berhasil dikirim ke DagsHub repository Anda.")