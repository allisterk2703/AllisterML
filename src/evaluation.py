import inspect
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression_model(name: str, y_test: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {name}: R² (↑): {r2:.2f}, RMSE (↓): {rmse:.2f}, MAE (↓): {mae:.2f}")
    return r2, rmse, mae


def evaluate_classification_model(
    name: str,
    classification_type: str,
    y_test: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series | None = None,
) -> dict:
    avg = "binary" if classification_type == "binary" else "macro"

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average=avg, zero_division=0))
    recall = float(recall_score(y_test, y_pred, average=avg, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average=avg, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    cls_report_str = classification_report(y_test, y_pred, digits=2)
    cls_report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    roc_auc = None
    if y_proba is not None:
        try:
            if classification_type == "binary":
                roc_auc = float(roc_auc_score(y_test, y_proba))
            else:
                roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    if roc_auc is not None:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {name}: F1-score (↑): {f1:.2f}, Recall (↑): {recall:.2f}, Precision (↑): {precision:.2f}, Accuracy (↑): {accuracy:.2f}, ROC AUC (↑): {roc_auc:.2f}")
    else:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {name}: F1-score (↑): {f1:.2f}, Recall (↑): {recall:.2f}, Precision (↑): {precision:.2f}, Accuracy (↑): {accuracy:.2f}")

    # Log compact report and CM for lecture rapide
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {name} — Classification report:\n{cls_report_str}")
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {name} — Confusion matrix:\n{cm}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report_str": cls_report_str,
        "classification_report": cls_report_dict,
    }
