
"""Train, tune, evaluate, and register the Tourism Package Prediction model."""

from pathlib import Path
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# 1. Paths and experiment configuration
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
DEPLOYMENT_DIR = PROJECT_DIR / "deployment"

TRAIN_PATH = ARTIFACT_DIR / "train.csv"
TEST_PATH = ARTIFACT_DIR / "test.csv"

TARGET = "ProdTaken"
RANDOM_STATE = 42

# Local MLflow tracking store. No external tracking server required.
MLFLOW_DIR = PROJECT_DIR / "mlruns"
mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
mlflow.set_experiment("Tourism_Package_Prediction")


# ---------------------------------------------------------
# 2. Load prepared training and testing datasets
# ---------------------------------------------------------

def load_splits():
    """Load and validate the train/test CSV files."""

    for path in (TRAIN_PATH, TEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required split not found: {path}\n"
                "Run prep.py before train.py."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if train_df.empty or test_df.empty:
        raise ValueError("Training and testing datasets must not be empty.")

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise ValueError(f"Both datasets must contain target '{TARGET}'.")

    if train_df[TARGET].isna().any() or test_df[TARGET].isna().any():
        raise ValueError(f"Target '{TARGET}' contains missing values.")

    if set(train_df.columns) != set(test_df.columns):
        raise ValueError("Training and testing columns do not match.")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training target must contain both classes 0 and 1.")

    if not set(y_test.unique()).issubset({0, 1}):
        raise ValueError("Testing target must contain only classes 0 and 1.")

    # Keep identical feature ordering in training and testing.
    X_test = X_test[X_train.columns]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------
# 3. Define preprocessing and model
# ---------------------------------------------------------

def build_pipeline(X):
    """Build a preprocessing pipeline followed by Random Forest."""

    numeric_features = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_features = [
        col for col in X.columns if col not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=1,
                ),
            ),
        ]
    )

    return pipeline


# ---------------------------------------------------------
# 4. Tune, evaluate, and log the best model
# ---------------------------------------------------------

def train_and_evaluate():
    """Tune the model, log the experiment, and save the best model."""

    warnings.filterwarnings(
        "ignore",
        message=".*feature names.*",
    )

    X_train, X_test, y_train, y_test = load_splits()
    pipeline = build_pipeline(X_train)

    # Tune the model using stratified cross-validation.
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting hyperparameter tuning...")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Parameter combinations: {len(list(
        __import__('sklearn').model_selection.ParameterGrid(param_grid)
    ))}")

    with mlflow.start_run(run_name="random_forest_grid_search") as run:
        # Record experiment configuration.
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("target", TARGET)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("scoring", "f1")
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("testing_rows", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("parameter_grid", json.dumps(param_grid))

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        # Evaluate once on the held-out test set.
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(y_test, y_pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, y_pred, zero_division=0)
            ),
            "f1_score": float(
                f1_score(y_test, y_pred, zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "best_cv_f1": float(search.best_score_),
        }

        # Log the selected hyperparameters and test/CV metrics.
        mlflow.log_params(
            {
                key.replace("classifier__", "rf_"): value
                for key, value in search.best_params_.items()
            }
        )
        mlflow.log_metrics(metrics)

        # Save detailed evaluation reports.
        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        report_path = DEPLOYMENT_DIR / "classification_report.json"
        metrics_path = DEPLOYMENT_DIR / "metrics.json"
        params_path = DEPLOYMENT_DIR / "best_params.json"
        confusion_path = DEPLOYMENT_DIR / "confusion_matrix.csv"
        cv_results_path = DEPLOYMENT_DIR / "cv_results.csv"

        report_path.write_text(json.dumps(report, indent=2))
        metrics_path.write_text(json.dumps(metrics, indent=2))
        params_path.write_text(
            json.dumps(search.best_params_, indent=2, default=str)
        )

        pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=[0, 1]),
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(confusion_path)

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.to_csv(cv_results_path, index=False)

        # Persist the entire pipeline, including preprocessing.
        model_path = DEPLOYMENT_DIR / "best_model.joblib"
        joblib.dump(best_model, model_path)

        metadata = {
            "model_type": "RandomForestClassifier",
            "target": TARGET,
            "model_file": model_path.name,
            "mlflow_run_id": run.info.run_id,
            "best_cv_f1": metrics["best_cv_f1"],
            "feature_columns": X_train.columns.tolist(),
            "best_params": search.best_params_,
            "metrics": metrics,
        }

        metadata_path = DEPLOYMENT_DIR / "model_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str)
        )

        # Store model and reports as MLflow artifacts.
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(params_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(cv_results_path))
        mlflow.log_artifact(str(metadata_path))

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.head(3),
        )

        print("\nBest parameters:")
        print(search.best_params_)

        print("\nHeld-out test metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nClassification report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print(f"\nSaved model: {model_path}")
        print(f"Saved metrics: {metrics_path}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"MLflow tracking directory: {MLFLOW_DIR}")

    print("\nTraining and evaluation completed successfully.")


if __name__ == "__main__":
    train_and_evaluate()
