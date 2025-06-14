# modelling_tuning.py

import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Konfigurasi MLflow (sama seperti sebelumnya)
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("submission_rifzki_hyperparameter_tuning")

# Muat data yang sudah diproses
train_df = pd.read_csv('lung_cancer_train_preprocessed.csv')
test_df = pd.read_csv('lung_cancer_test_preprocessed.csv')

X_train = train_df.drop('LUNG_CANCER', axis=1)
y_train = train_df['LUNG_CANCER']
X_test = test_df.drop('LUNG_CANCER', axis=1)
y_test = test_df['LUNG_CANCER']

# Input example untuk model signature
input_example = X_train.head(5)

# 1. Tentukan rentang hyperparameter untuk diuji
list_n_neighbors = [3, 5, 7, 9, 11]
algorithm = 'auto'

print("Memulai eksperimen Hyperparameter Tuning untuk KNN...")

# 2. Lakukan iterasi untuk setiap nilai hyperparameter
for n in list_n_neighbors:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"KNN_Tuning_n_{n}_{timestamp}"
    
    # 3. Mulai run baru untuk setiap iterasi
    with mlflow.start_run(run_name=run_name):
        # Log parameter yang digunakan di run ini
        mlflow.log_param("n_neighbors", n)
        mlflow.log_param("algorithm", algorithm)

        # Latih model dengan hyperparameter saat ini
        model = KNeighborsClassifier(n_neighbors=n, algorithm=algorithm)
        model.fit(X_train, y_train)

        # Log model sebagai artefak
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Evaluasi dan log metrik
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        print(f"Run '{run_name}' selesai.")
        print(f"  n_neighbors: {n}")
        print(f"  Akurasi: {accuracy}\n")

print("Eksperimen Hyperparameter Tuning selesai.")