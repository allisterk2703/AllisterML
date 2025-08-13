import inspect
import logging
import os
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src.config_loading import Config
from src.evaluation import evaluate_classification_model, evaluate_regression_model


def train_regression_models(
    CFG: Config,
    X_train_scaled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    results = []

    models = {
        model: model_class
        for model, model_class in {
            "XGBRegressor": XGBRegressor(random_state=42),
            "LGBMRegressor": LGBMRegressor(random_state=42, verbose=-1),
            "RandomForestRegressor": RandomForestRegressor(random_state=42),
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "SVR": SVR(),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        }.items()
        if CFG.TRAINING_MODELS[CFG.PROBLEM_TYPE.upper()].get(model.upper(), 0) == 1
    }
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Models to train: {list(models.keys())}")

    for name, base_model in models.items():
        run_name = f"{name}-{CFG.DATASET_NAME}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        ctx = mlflow.start_run(run_name=run_name) if CFG.LOG_IN_MLFLOW else nullcontext()
        with ctx:
            if CFG.LOG_IN_MLFLOW:
                mlflow.set_tags(
                    {
                        "dataset": CFG.DATASET_NAME,
                        "problem_type": CFG.PROBLEM_TYPE,
                        "stage": "train",
                        "model_name": name,
                    }
                )

            model = base_model
            best_params = {}

            grid = CFG.PARAM_GRIDS_REGRESSION.get(name)
            if CFG.TRAINING_GRIDSEARCH and grid:
                logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] GridSearch for {name} with grid: {grid}")
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_param("gridsearch_enabled", CFG.TRAINING_GRIDSEARCH)
                    mlflow.log_param("cv_folds", CFG.TRAINING_CV_FOLDS)

                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=grid,
                    cv=CFG.TRAINING_CV_FOLDS,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
                grid_search.fit(X_train_scaled, y_train_resampled)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
            else:
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_param("gridsearch_enabled", False)
                model.fit(X_train_scaled, y_train_resampled)

            # Log des hyperparamètres finaux
            try:
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_params({f"final__{k}": v for k, v in model.get_params().items()})
            except Exception as e:
                logging.info(f"Could not log model params for {name}: {e}")

            # Évaluation
            y_pred = model.predict(X_test_scaled)
            r2, rmse, mae = evaluate_regression_model(name, y_test, y_pred)

            if CFG.LOG_IN_MLFLOW:
                mlflow.log_metrics(
                    {
                        "r2": float(r2),
                        "rmse": float(rmse),
                        "mae": float(mae),
                    }
                )
                try:
                    mlflow.log_param("n_features", X_train_scaled.shape[1])
                except Exception:
                    pass
                mlflow.sklearn.log_model(model, name="model", registered_model_name=None)

            # Récap local
            results.append(
                {
                    "model": name,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                }
            )

    return results


def train_classification_models(
    CFG: Config,
    X_train_scaled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    results = []

    models = {
        model: model_class
        for model, model_class in {
            "XGBClassifier": XGBClassifier(random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "LGBMClassifier": LGBMClassifier(random_state=42, verbose=-1),
            "CatBoostClassifier": CatBoostClassifier(verbose=0, random_state=42),
            "RandomForestClassifier": RandomForestClassifier(random_state=42),
            "SVC": SVC(random_state=42),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
            "NaiveBayesClassifier": GaussianNB(),
        }.items()
        if CFG.TRAINING_MODELS[CFG.PROBLEM_TYPE.upper()].get(model.upper(), 0) == 1
    }
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Models to train: {list(models.keys())}")

    for name, base_model in models.items():
        run_name = f"{name}-{CFG.DATASET_NAME}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ctx = mlflow.start_run(run_name=run_name) if CFG.LOG_IN_MLFLOW else nullcontext()

        with ctx:
            if CFG.LOG_IN_MLFLOW:
                mlflow.set_tags(
                    {
                        "dataset": CFG.DATASET_NAME,
                        "problem_type": CFG.PROBLEM_TYPE,
                        "classification_type": CFG.CLASSIFICATION_TYPE,
                        "stage": "train",
                        "model_name": name,
                    }
                )

            model = base_model
            best_params = {}

            # GridSearch si demandé et si un grid existe
            if CFG.TRAINING_GRIDSEARCH and name in CFG.PARAM_GRIDS_CLASSIFICATION:
                grid = CFG.PARAM_GRIDS_CLASSIFICATION[name]
                logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] GridSearch for {name} with grid: {grid}")
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_param("gridsearch_enabled", CFG.TRAINING_GRIDSEARCH)
                    mlflow.log_param("cv_folds", CFG.TRAINING_CV_FOLDS)

                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=grid,
                    cv=CFG.TRAINING_CV_FOLDS,
                    scoring="accuracy",
                    n_jobs=-1,
                )
                grid_search.fit(X_train_scaled, y_train_resampled)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
            else:
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_param("gridsearch_enabled", False)
                model.fit(X_train_scaled, y_train_resampled)

            # Log des hyperparams finaux
            try:
                if CFG.LOG_IN_MLFLOW:
                    mlflow.log_params({f"final__{k}": v for k, v in model.get_params().items()})
            except Exception as e:
                logging.info(f"Could not log model params for {name}: {e}")

            # Prédictions
            y_pred = model.predict(X_test_scaled)
            y_proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test_scaled)
                if CFG.CLASSIFICATION_TYPE == "binary":
                    y_proba = proba[:, 1]
                else:
                    y_proba = proba
            elif hasattr(model, "decision_function") and CFG.CLASSIFICATION_TYPE == "binary":
                y_proba = model.decision_function(X_test_scaled)

            # Évaluation
            metrics = evaluate_classification_model(name, CFG.CLASSIFICATION_TYPE, y_test, y_pred, y_proba)

            if CFG.LOG_IN_MLFLOW:
                # métriques coeur
                mlflow.log_metrics(
                    {
                        "accuracy": float(metrics["accuracy"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        **({"roc_auc": float(metrics["roc_auc"])} if metrics.get("roc_auc") is not None else {}),
                    }
                )
                # infos features
                try:
                    mlflow.log_param("n_features", X_train_scaled.shape[1])
                except Exception:
                    pass
                # modèle
                mlflow.sklearn.log_model(model, name="model", registered_model_name=None)

            results.append(
                {
                    "model": name,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "roc_auc": metrics.get("roc_auc"),
                }
            )

    return results


def results_to_df(CFG: Config, results: list[dict]) -> pd.DataFrame:
    results_df = pd.DataFrame(results).set_index("model").round(3)
    if CFG.PROBLEM_TYPE == "regression":
        results_df.sort_values(by="r2", ascending=False, inplace=True)
    elif CFG.PROBLEM_TYPE == "classification":
        results_df.sort_values(by="f1", ascending=False, inplace=True)

    if CFG.SAVE_RESULTS_TO_CSV:
        results_df.to_csv(CFG.DATA_FOLDER_PATH / "results.csv")

    return results_df


def delete_catboost_info() -> None:
    if os.path.exists("catboost_info"):
        shutil.rmtree("catboost_info")
    return


def apply_models(
    CFG: Config,
    X_train_scaled: pd.DataFrame,
    y_train_resampled: pd.Series,
    X_test_scaled: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Training {CFG.PROBLEM_TYPE} models")

    if CFG.PROBLEM_TYPE == "regression":
        results = train_regression_models(CFG, X_train_scaled, y_train_resampled, X_test_scaled, y_test)
    elif CFG.PROBLEM_TYPE == "classification":
        results = train_classification_models(CFG, X_train_scaled, y_train_resampled, X_test_scaled, y_test)
    else:
        raise ValueError(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] PROBLEM_TYPE must be 'regression' or 'classification', received: {CFG.PROBLEM_TYPE}")

    results_df = results_to_df(CFG, results)

    delete_catboost_info()

    return results_df
