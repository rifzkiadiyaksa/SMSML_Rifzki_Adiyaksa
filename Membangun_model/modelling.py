import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split # Meskipun data sudah split, ini untuk jaga-jaga
from datetime import datetime
import joblib # Untuk memuat scaler jika dibutuhkan

# Konfigurasi MLflow Tracking
# Untuk saat ini, kita akan menggunakan MLflow Tracking secara lokal.
# Nanti untuk poin Advance, kita akan pindah ke DagsHub.
mlflow.set_tracking_uri("http://127.0.0.1:5000/") # Pastikan ini mengarah ke server MLflow UI lokal Anda

# Set nama eksperimen Anda
mlflow.set_experiment("submission_rifzki_basic_model")

# Muat data yang sudah diproses dari Kriteria 1
# Sesuaikan path jika folder `preprocessing` tidak berada di level yang sama
train_df = pd.read_csv('lung_cancer_train_preprocessed.csv')
test_df = pd.read_csv('lung_cancer_test_preprocessed.csv')

X_train = train_df.drop('LUNG_CANCER', axis=1)
y_train = train_df['LUNG_CANCER']
X_test = test_df.drop('LUNG_CANCER', axis=1)
y_test = test_df['LUNG_CANCER']

# Anda bisa memuat scaler juga jika model Anda membutuhkannya saat inferensi
# scaler = joblib.load('scaler.joblib')

# Contoh input untuk logging model
input_example = X_train.head(5)

# Buat nama run yang unik dengan timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"KNN_Modelling_Basic_{timestamp}"

with mlflow.start_run(run_name=run_name):
    # Log parameter model
    n_neighbors = 5
    algorithm = 'auto'
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("algorithm", algorithm)

    # Nonaktifkan autologging jika Anda ingin kontrol penuh atas logging manual
    # mlflow.autolog(disable=True)

    # Latih model (menggunakan KNeighborsClassifier seperti di referensi)
    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    model.fit(X_train, y_train)

    # Log model sebagai artefak
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )

    # Log metrik evaluasi
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Eksperimen 'KNN_Modelling_Basic_{timestamp}' selesai dan telah dicatat di MLflow.")
    print(f"Akurasi model: {accuracy}")