# modelling.py (Versi DagsHub)

import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub  # Import library DagsHub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# --- Inisialisasi DagsHub ---
# Baris ini akan secara otomatis mengkonfigurasi MLflow untuk terhubung ke DagsHub
dagshub.init(repo_owner='rifzkiadiyaksa', repo_name='SMSML_Rifzki_Adiyaksa', mlflow=True)
print("Koneksi ke DagsHub MLflow Tracking Server telah diinisialisasi.")

# Nama eksperimen Anda
experiment_name = "Lung_Cancer_Prediction_Basic"
mlflow.set_experiment(experiment_name)
print(f"Eksperimen diatur ke '{experiment_name}'")

# Memuat Data Latih dan Uji
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

# Memulai Run MLflow
with mlflow.start_run() as run:
    print(f"\nMemulai run baru: {run.info.run_id}")

    # Melatih Model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    print("Model Logistic Regression telah dilatih.")

    # Membuat prediksi pada data uji
    y_pred = model.predict(X_test)
    print("Prediksi pada data uji telah dibuat.")

    # Mencatat Parameter
    params = model.get_params()
    mlflow.log_params(params)
    print("Parameter model telah dicatat:", params)

    # Mencatat Metrik
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

    # Mencatat Model (Artefak)
    mlflow.sklearn.log_model(model, "model")
    print("Model telah dicatat sebagai artefak.")

    print("\nRun MLflow selesai dan dikirim ke DagsHub.")