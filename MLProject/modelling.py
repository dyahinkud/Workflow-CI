import os
import shutil
import warnings
import dagshub
import mlflow
import pandas as pd

from mlflow import sklearn as mlflow_sklearn

from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

dagshub.init(repo_owner="dyahinkud", repo_name="Membangun_Model", mlflow=True)

data = pd.read_csv("./data_clean.csv")
data = data.dropna(subset=["cleaned_text"])

X = data["cleaned_text"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# Ensure y_train and y_test are Series (1D) for sklearn and MLflow
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# Suppress pickle warning from sklearn
warnings.filterwarnings(
    "ignore",
    message="Saving scikit-learn models in the pickle or cloudpickle format requires exercising caution",
)

# Suppress DataConversionWarning from sklearn
warnings.filterwarnings("ignore", category=DataConversionWarning)


pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression()),
    ]
)

param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__solver": ["lbfgs"],
    "clf__class_weight": ["balanced", None],
    "clf__max_iter": [500, 1000],
}

with mlflow.start_run(nested=False):
    mlflow_sklearn.autolog()

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy_manual", float(acc))
    mlflow.log_params(grid.best_params_)

    print("Best Params:", grid.best_params_)
    print("Accuracy:", acc)
    report = classification_report(y_test, y_pred, output_dict=False)
    print("Classification Report:\n", report)

    mlflow.log_text(str(report), "classification_report.txt")

    # Workaround for DagsHub MLflow artifact upload issue
    if os.path.exists("best_model_local"):
        shutil.rmtree("best_model_local")

    mlflow_sklearn.save_model(best_model, "best_model_local")
    mlflow.log_artifacts("best_model_local", "model")

    # Save run_id for GitHub Actions to use
    with open("run_id.txt", "w") as f:
        f.write(mlflow.active_run().info.run_id)
