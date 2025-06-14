import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 1. Mengatur URI Pelacakan MLflow
# MLflow akan menyimpan output eksperimen di folder 'mlruns' di direktori yang sama dengan skrip ini.
# Pastikan Anda sudah membuat folder 'mlruns' atau MLflow akan membuatnya secara otomatis.
mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), "mlruns"))

# Nama eksperimen Anda
experiment_name = "Lung_Cancer_Prediction_Basic"
mlflow.set_experiment(experiment_name)

print("MLflow tracking URI diatur ke 'mlruns' folder lokal.")
print(f"Eksperimen diatur ke '{experiment_name}'")

# 2. Memuat Data Latih dan Uji
try:
    train_data = pd.read_csv('lung_cancer_train_preprocessed.csv')
    test_data = pd.read_csv('lung_cancer_test_preprocessed.csv')
    print("Data latih dan uji berhasil dimuat.")
except FileNotFoundError as e:
    print(f"Error: {e}. Pastikan file CSV berada di folder yang sama.")
    exit()

# Memisahkan fitur (X) dan target (y)
X_train = train_data.drop('LUNG_CANCER', axis=1)
y_train = train_data['LUNG_CANCER']
X_test = test_data.drop('LUNG_CANCER', axis=1)
y_test = test_data['LUNG_CANCER']
print("Fitur dan target telah dipisahkan.")

# 3. Memulai Run MLflow
# Semua pencatatan (logging) terjadi di dalam blok 'with' ini.
with mlflow.start_run() as run:
    print(f"\nMemulai run baru: {run.info.run_id}")

    # 4. Melatih Model
    # Kita akan menggunakan Logistic Regression sebagai model dasar.
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    print("Model Logistic Regression telah dilatih.")

    # Membuat prediksi pada data uji
    y_pred = model.predict(X_test)
    print("Prediksi pada data uji telah dibuat.")

    # 5. Mencatat Parameter (Logging Parameters)
    # Ini adalah 'hyperparameters' dari model kita.
    params = model.get_params()
    mlflow.log_params(params)
    print("Parameter model telah dicatat:", params)

    # 6. Mencatat Metrik (Logging Metrics)
    # Ini adalah hasil evaluasi kinerja model.
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    print("Metrik evaluasi telah dicatat.")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # 7. Mencatat Model (Artefak)
    # Ini menyimpan model yang telah dilatih sebagai sebuah artefak di dalam run MLflow.
    # Ini adalah syarat PENTING untuk kelulusan.
    mlflow.sklearn.log_model(model, "model")
    print("Model telah dicatat sebagai artefak.")

    print("\nRun MLflow selesai.")