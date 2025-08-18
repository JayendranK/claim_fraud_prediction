import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")


# --------------------------
# Configurations
# --------------------------
MODEL_DIR = "models"
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "claim-fraud-detection"
MODEL_NAME = "claim-fraud-detection-best" # if using registry


def get_models():
    """Return dictionary of candidate models."""

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Bagging": BaggingClassifier(random_state=42)
    }

    return models

def train_and_log_all(X_train, X_test, y_train, y_test, preprocessor):
    """Train and log all models to MLflow, save & register best model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    models = get_models()
    results = {}
    run_ids = {}

    for model_name, clf in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            pipeline = Pipeline(steps=[
                ("preprocessing", preprocessor),
                ("classifier", clf)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, pos_label='Yes')
            rec = recall_score(y_test, y_pred, pos_label='Yes')

            # Log to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            })

            # Save model locally
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(pipeline, f"{MODEL_DIR}/{model_name}.joblib")

            # Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model"
            )

            # Store for best model comparison
            results[model_name] = {
                "accuracy": acc,
                "pipeline": pipeline
            }
            run_ids[model_name] = run.info.run_id

            print(f"✅ {model_name} — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    # Select best model
    best_model_name = max(results, key=lambda m: results[m]['accuracy'])
    best_pipeline = results[best_model_name]['pipeline']
    best_run_id = run_ids[best_model_name]

    # Save best model locally
    joblib.dump(best_pipeline, f"{MODEL_DIR}/best_model.joblib")
    print(f"🏆 Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

    # Register best model in MLflow Model Registry
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )
    print(f"📦 Registered best model '{best_model_name}' as '{MODEL_NAME}'")