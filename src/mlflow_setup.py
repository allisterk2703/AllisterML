import inspect
import logging
import os
import shutil
from pathlib import Path

import mlflow

from src.config_loading import Config


def delete_default_mlflow_experiment():
    try:
        default_experiment_path = Path("mlruns/0")
        if default_experiment_path.exists() and default_experiment_path.is_dir():
            shutil.rmtree(default_experiment_path)
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Default MLflow experiment deleted.")
        else:
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Default MLflow experiment folder not found.")
    except Exception as e:
        logging.error(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Failed to delete default MLflow experiment: {e}")


def setup_mlflow(CFG: Config, experiment_name: str):
    mlruns_dir = CFG.DATA_FOLDER_PATH / "mlruns"

    # Créer le répertoire mlruns sous DATA_PATH s'il n'existe pas
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    # Configurer l'environnement de MLflow pour utiliser le répertoire mlruns dans DATA_PATH
    os.environ["MLFLOW_TRACKING_URI"] = str(mlruns_dir)

    try:
        # Définir l'expérience MLflow
        mlflow.set_experiment(str(mlruns_dir / experiment_name))
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] MLflow experiment '{experiment_name}' set up at {mlruns_dir}")
        delete_default_mlflow_experiment()
    except Exception as e:
        logging.error(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Error setting up MLflow experiment: {e}")
