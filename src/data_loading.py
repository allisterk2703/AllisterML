import inspect
import logging
import os
from pathlib import Path

import pandas as pd

from src.config_loading import Config


def import_data(CFG: Config) -> pd.DataFrame:
    for file in os.listdir(CFG.DATA_FOLDER_PATH):
        if file.split(".")[0] == "data":
            if file.endswith(".csv"):
                df = pd.read_csv(Path(CFG.DATA_FOLDER_PATH) / file)
            elif file.endswith((".xlsx", ".xls")):
                df = pd.read_excel(Path(CFG.DATA_FOLDER_PATH) / file)
            elif file.endswith((".parquet", ".pq")):
                df = pd.read_parquet(Path(CFG.DATA_FOLDER_PATH) / file)
            elif file.endswith(".json"):
                df = pd.read_json(Path(CFG.DATA_FOLDER_PATH) / file)
            else:
                raise ValueError(f"File type not supported: '{file}'")

            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {file.split('.')[-1].upper()} dataset imported from '{CFG.DATASET_NAME}/{file}'")
            return df

    raise FileNotFoundError(f"No dataset named '{CFG.DATASET_NAME}' found in {CFG.DATA_FOLDER_PATH}")
