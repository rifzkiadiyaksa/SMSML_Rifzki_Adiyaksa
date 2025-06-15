import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import dagshub
import json

# --- KOREKSI KRITIKAL: Inisialisasi DagsHub MLflow tracking dengan kredensial rifzkiadiyaksa ---
dagshub.init(repo_owner='rifzkiadiyaksa', repo_name='SMSML_Rifzki_Adiyaksa', mlflow=True)

# --- KOREKSI KRITIKAL: Set MLflow tracking URI ke DagsHub rifzkiadiyaksa ---
mlflow.set_tracking_uri("https://dagshub.com/rifzkiadiyaksa/SMSML_Rifzki_Adiyaksa.mlflow")

# Nama eksperimen MLflow
mlflow.set_experiment("Lung Cancer Prediction Model")

# Muat data yang sudah diproses
# Pastikan file-file ini sudah disalin ke /content/ sebelum menjalankan script
train_data_path = "/content/lung_cancer_train_preprocessed.csv"
test_data_path = "/content/lung_cancer_test_preprocessed.csv"

try:
    X_train = pd.read_csv(train_data_path).drop("lung_cancer", axis=1)
    y_train = pd.read_csv(train_data_path)["lung_cancer"]
    X_test = pd.read_csv(test_data_path).drop("lung_cancer", axis=1)
    y_test = pd.read_csv(test_data_path)["lung_cancer"]
    print(f"Data train dari '{train_data_path}' berhasil dimuat.")
    print(f"Data test dari '{test_data_path}' berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: File data tidak ditemukan. Pastikan file ada di '{train_data_path}' dan '{test_data_path}'.")
    raise e

# Contoh input untuk logging model (ambil 5 sampel dari X_train)
input_example = X_train.sample(5)

# Buat nama run unik dengan timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"KNN_Modelling_ManualLog_{timestamp}"

with mlflow.start_run(run_name=run_name):
    mlflow.autolog(disable=True) # Nonaktifkan autolog untuk manual logging

    # Definisikan parameter model
    n_neighbors = 5
    algorithm = 'auto'
    
    # Log parameter secara manual
    mlflow.log_param("n_neighbors", n_neighbors)
    mlflow.log_param("algorithm", algorithm)
    print(f"Parameter: n_neighbors={n_neighbors}, algorithm={algorithm} telah dilog.")

    # Latih model K-Nearest Neighbors
    model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
    model.fit(X_train, y_train)
    print("Model KNN berhasil dilatih.")

    # Lakukan prediksi
    y_pred = model.predict(X_test)

    # Hitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Log metrik secara manual
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    print(f"Metrik: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f} telah dilog.")

    # Log model ke MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="knn_model", # Nama artefak model
        input_example=input_example,
        registered_model_name="KNNLungCancerModel" # Daftarkan model jika ingin di Registry
    )
    print("Model berhasil dilog ke MLflow.")

    # Log artefak tambahan (misalnya daftar kolom fitur)
    with open("feature_columns.txt", "w") as f:
        json.dumps(X_train.columns.tolist())
    mlflow.log_artifact("feature_columns.txt")
    print("Daftar kolom fitur telah dilog sebagai artefak.")

print("MLflow Run selesai. Cek DagsHub untuk detailnya.")