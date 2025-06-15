import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
import json
from datetime import datetime
import dagshub

# Inisialisasi DagsHub untuk integrasi dengan MLflow
dagshub.init(repo_owner='rifdahhhh', repo_name='lung-cancer-submission', mlflow=True)

# Mengatur URI pelacakan MLflow ke repositori DagsHub Anda
mlflow.set_tracking_uri("https://dagshub.com/rifdahhhh/lung-cancer-submission.mlflow")

# Membaca dataset
data = pd.read_csv("lung_cancer_clean.csv")

# Memisahkan fitur dan target, lalu membaginya menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("lung_cancer", axis=1),
    data["lung_cancer"],
    test_size=0.2,
    random_state=42
)

# Membuat nama unik untuk setiap run
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"KNN_Tuning_{timestamp}"

with mlflow.start_run(run_name=run_name) as run:
    # Membuat pipeline: standarisasi data diikuti oleh pemodelan KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Menentukan rentang hiperparameter yang akan diuji
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski']
    }

    # Mencari parameter terbaik menggunakan GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=param_grid,
                               cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Menggunakan model terbaik untuk prediksi
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log parameter dan metrik terbaik ke MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_class_0", report['0']['precision'])
    mlflow.log_metric("recall_class_0", report['0']['recall'])
    mlflow.log_metric("precision_class_1", report['1']['precision'])
    mlflow.log_metric("recall_class_1", report['1']['recall'])

    # Membuat dan mencatat confusion matrix sebagai gambar
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('KNN Confusion Matrix (Tuned)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plot_path = "training_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    mlflow.log_artifact(plot_path)

    # Menyimpan model terbaik dengan signature
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(best_model, "best_knn_model", signature=signature)

    # Menyimpan classification report sebagai file JSON
    with open("metric_info.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("metric_info.json")

    print(f"Best model found and logged with accuracy: {accuracy}")
    print(f"Run '{run_name}' logged to MLflow.")