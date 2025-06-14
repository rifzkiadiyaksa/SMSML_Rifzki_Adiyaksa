import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Mengatur URI pelacakan ke folder lokal yang sama
mlflow.set_tracking_uri("file://" + os.path.join(os.getcwd(), "mlruns"))
experiment_name = "Lung_Cancer_Prediction_Tuning" # Nama eksperimen baru untuk tuning
mlflow.set_experiment(experiment_name)

print("MLflow tracking URI diatur ke 'mlruns' folder lokal.")
print(f"Eksperimen diatur ke '{experiment_name}'")

# Memuat data yang sudah diproses
train_data = pd.read_csv('lung_cancer_train_preprocessed.csv')
test_data = pd.read_csv('lung_cancer_test_preprocessed.csv')

X_train = train_data.drop('LUNG_CANCER', axis=1)
y_train = train_data['LUNG_CANCER']
X_test = test_data.drop('LUNG_CANCER', axis=1)
y_test = test_data['LUNG_CANCER']

# Mendefinisikan model dan grid parameter untuk dicoba
model = LogisticRegression(max_iter=1000, random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}

# Menggunakan GridSearchCV untuk menemukan parameter terbaik
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Memulai run utama (parent run) untuk GridSearchCV
with mlflow.start_run(run_name="GridSearchCV_Parent_Run") as parent_run:
    print(f"\nMemulai Parent Run: {parent_run.info.run_id}")

    # Latih grid search
    grid_search.fit(X_train, y_train)
    print("GridSearchCV selesai.")

    # Log parameter terbaik dan skor terbaik dari parent run
    mlflow.log_param("best_params", grid_search.best_params_)
    mlflow.log_metric("best_accuracy", grid_search.best_score_)
    print(f"Parameter terbaik: {grid_search.best_params_}")
    print(f"Skor akurasi terbaik dari CV: {grid_search.best_score_:.4f}")

    # Mencatat setiap kombinasi sebagai nested run (child run)
    print("\nMencatat setiap percobaan sebagai nested run...")
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        with mlflow.start_run(run_name=f"Child_Run_{i}", nested=True) as child_run:
            # Log parameter untuk run ini
            mlflow.log_params(cv_results['params'][i])
            # Log skor validasi rata-rata
            mlflow.log_metric("mean_test_score", cv_results['mean_test_score'][i])
            print(f"  Run {i}: Params: {cv_results['params'][i]}, Score: {cv_results['mean_test_score'][i]:.4f}")

    # Simpan model terbaik sebagai artefak
    best_model = grid_search.best_estimator_
    mlflow.sklearn.log_model(best_model, "best_model")
    print("\nModel terbaik telah dicatat sebagai artefak.")

print("\nSemua run MLflow untuk tuning telah selesai.")