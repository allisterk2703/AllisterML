import inspect
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config_loading import Config


def get_dimensions(df: pd.DataFrame) -> None:
    nb_row, nb_col = df.shape
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Number of rows: {nb_row}")
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Number of columns: {nb_col}")
    return


def reformat_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    logging.info(
        f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Column names reformated: {list(df.columns)}"
    )
    return df


def rename_columns(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if CFG.RENAME_COLUMNS:
        for col in CFG.RENAME_COLUMNS:
            new_col = col.lower().replace(" ", "_")
            if new_col in df.columns:
                df = df.rename(columns={new_col: CFG.RENAME_COLUMNS[col]})
                logging.info(
                    f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Column '{new_col}' renamed to '{CFG.RENAME_COLUMNS[col]}'"
                )
            else:
                logging.info(
                    f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Column '{new_col}' not found in dataframe"
                )
    else:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'rename_columns' skipped because `COLUMNS.RENAME: {{}}`"
        )
    return df


def replace_elements_by_nan(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if CFG.ELEMENTS_TO_REPLACE_BY_NAN:
        for elem in CFG.ELEMENTS_TO_REPLACE_BY_NAN:
            before_count = (df == elem).sum().sum()
            df = df.replace(elem, np.nan)
            after_count = (df == elem).sum().sum()
            replaced_count = before_count - after_count
            logging.info(
                f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Replaced '{elem}' by NaN ({replaced_count} occurrences)"
            )
    else:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'replace_elements_by_nan' skipped because `CLEANING.REPLACE_BY_NAN: {{}}`"
        )
    return df


def unique_object_values(df: pd.DataFrame) -> None:
    func_name = f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()]"

    object_cols = df.select_dtypes(include=["object"]).columns
    if object_cols.empty:
        logging.info(f"{func_name} No object columns found")
        return

    info_msgs = []
    warn_msgs = []

    for col in object_cols:
        unique_vals = df[col].unique()
        msg = f"{col}: {unique_vals}"
        if df[col].nunique() == 1:
            warn_msgs.append(msg)
        else:
            info_msgs.append(msg)

    if info_msgs:
        logging.info(f"{func_name}\n" + "\n".join(info_msgs))
    if warn_msgs:
        logging.warning(f"{func_name} (Constant columns)\n" + "\n".join(warn_msgs))


def replace_elements_by_something(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if CFG.REPLACE_ELEMENTS_BY_SOMETHING:
        for col, replacements_dict in CFG.REPLACE_ELEMENTS_BY_SOMETHING.items():
            logging.info(
                f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Remplacement dans '{col}' : {replacements_dict}"
            )
            try:
                df[col] = df[col].replace(replacements_dict)
            except Exception as e:
                logging.error(
                    f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Erreur lors du remplacement dans '{col}': {e}"
                )
    else:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'replace_elements_by_something' skipped because `CLEANING.REPLACE_MAP: {{}}`"
        )

    return df


def replace_in_specific_columns(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if CFG.REPLACE_IN_SPECIFIC_COLUMNS:
        for col, replacements_dict in CFG.REPLACE_IN_SPECIFIC_COLUMNS.items():
            logging.info(
                f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Remplacement dans '{col}' : {replacements_dict}"
            )
            df[col] = df[col].replace(replacements_dict)
    return df


def nb_unique_values(CFG: Config, df: pd.DataFrame) -> None:
    object_cols = df.select_dtypes(include=["object"]).columns

    if object_cols.empty:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] All columns are numerical")
        return

    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Categorical features:")
    for col in object_cols:
        if col != CFG.TARGET_VARIABLE:
            if df[col].nunique() == 1:
                logging.warning(
                    f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] -  {col}: {df[col].nunique()}"
                )
            else:
                logging.info(
                    f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] - {col}: {df[col].nunique()}"
                )

    if CFG.TARGET_VARIABLE in object_cols:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Target variable: {CFG.TARGET_VARIABLE} ({df[CFG.TARGET_VARIABLE].nunique()} unique values)"
        )
    return


def drop_unnecessary_columns(CFG: Config, df: pd.DataFrame, after_transformations: bool = False) -> pd.DataFrame:
    if not after_transformations:
        if CFG.UNNECESSARY_COLUMNS:
            for col in CFG.UNNECESSARY_COLUMNS:
                if col in df.columns:
                    percent_missing = (df[col].isna().sum() / len(df)) * 100
                    unique_count = df[col].nunique(dropna=True)
                    df = df.drop(columns=col)
                    logging.info(
                        f"Unnecessary column '{col}' dropped ({percent_missing:.2f}% missing, {unique_count} unique values)"
                    )
                else:
                    logging.info(f"Column '{col}' not found in dataframe")
        else:
            logging.info(
                f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'drop_unnecessary_columns' skipped because `COLUMNS.DROP: {{}}`"
            )
    elif after_transformations:
        if CFG.UNNECESSARY_COLUMNS_AFTER_TRANSFORMATIONS:
            for col in CFG.UNNECESSARY_COLUMNS_AFTER_TRANSFORMATIONS:
                if col in df.columns:
                    df = df.drop(columns=col)
                    logging.info(f"Unnecessary column '{col}' dropped after transformations")
        else:
            logging.info(
                f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'drop_unnecessary_columns' skipped because `COLUMNS.DROP_AFTER_TRANSFORMATIONS: {{}}`"
            )
    return df


def reorder_columns(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    int_cols = sorted([col for col in df.select_dtypes(include=["int", "int64"]).columns if col != CFG.TARGET_VARIABLE])
    float_cols = sorted(
        [col for col in df.select_dtypes(include=["float", "float64"]).columns if col != CFG.TARGET_VARIABLE]
    )
    object_cols = sorted(
        [col for col in df.select_dtypes(include=["object", "string"]).columns if col != CFG.TARGET_VARIABLE]
    )

    reordered_columns = int_cols + float_cols + object_cols + [CFG.TARGET_VARIABLE]
    df = df[reordered_columns]
    logging.info(
        f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Columns reordered: {reordered_columns}"
    )
    return df


def show_data_types(df: pd.DataFrame) -> None:
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()]\n{df.dtypes}")
    return


def show_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    null_counts = df.isnull().sum()
    na_counts = df.isna().sum()
    if null_counts.sum() == 0 and na_counts.sum() == 0:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] No missing values found")
        return
    else:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Missing values found:")
        missing_df = pd.DataFrame(
            {
                "Column": df.columns,
                "NULL": null_counts,
                "NaN": na_counts,
                "% Missing": (na_counts / len(df)) * 100,
            }
        )

        missing_df = missing_df[missing_df["% Missing"] > 0]
        missing_df = (
            missing_df.sort_values(by="% Missing", ascending=False).reset_index(drop=True).set_index("Column").round(2)
        )

        return missing_df


def deal_missing_values(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if CFG.MISSING_VALUES_IMPUTATION_STRATEGY:
        for col, strategy in CFG.MISSING_VALUES_IMPUTATION_STRATEGY.items():
            if col not in df.columns:
                logging.info(f"Colonne '{col}' introuvable, on ignore.")
                continue

            if isinstance(strategy, str):
                logging.info(f"Imputing '{col}' with '{strategy}'")
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].astype(float).mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].astype(float).median())
                elif strategy == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strategy == "drop":
                    before_rows = len(df)
                    df = df.dropna(subset=[col])
                    after_rows = len(df)
                    logging.info(f"Dropped {before_rows - after_rows} rows due to NaN in '{col}'")
                else:
                    raise ValueError(f"Unknown imputation strategy: {strategy}")
            elif isinstance(strategy, (int, float)):
                logging.info(f"Imputing '{col}' with 'constant={strategy}'")
                df[col] = df[col].fillna(strategy)
    else:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'deal_missing_values' skipped because `CLEANING.IMPUTATION: {{}}`"
        )
    return df


def drop_na_target_variable(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    previous_nb_rows = df.shape[0]
    df = df.dropna(subset=[CFG.TARGET_VARIABLE])
    new_nb_rows = df.shape[0]
    logging.info(
        f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {previous_nb_rows - new_nb_rows} rows dropped because of missing values in target column '{CFG.TARGET_VARIABLE}'"
    )
    return df


def check_missing_values(df: pd.DataFrame) -> str:
    total_missing = int(df.isnull().sum().sum())

    if total_missing > 0:
        messages = []
        for col in df.columns[df.isnull().any()]:
            count = df[col].isnull().sum()
            msg = f"{col} ({df[col].dtype}): {count} missing values found"
            logging.info(msg)
            messages.append(msg)
        return "\n".join(messages)
    else:
        msg = "No missing values found"
        logging.info(msg)
        return msg


def convert_data_type(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if CFG.DATA_TYPE_CONVERSION:
        for col, dtype in CFG.DATA_TYPE_CONVERSION.items():
            try:
                if dtype == "datetime":
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype)
                logging.info(
                    f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Converted '{col}' to '{dtype}'"
                )
            except Exception as e:
                logging.error(
                    f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Error converting '{col}' to '{dtype}': {e}"
                )
                raise e
    else:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'convert_data_type' skipped because `CLEANING.TYPE_CONVERSION: {{}}`"
        )
    return df


def plot_boxplot(CFG: Config, df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        plt.figure(figsize=(5, 1.8))
        sns.boxplot(x=df[col], color="skyblue")
        if col == CFG.TARGET_VARIABLE:
            plt.title(f"Boxplot – {col} (Target)")
        else:
            plt.title(f"Boxplot – {col} (Feature)")
        plt.tight_layout()
        plt.show()
    return


def deal_outliers(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if CFG.DEAL_OUTLIERS_STRATEGY:
        for strategy_dict in CFG.DEAL_OUTLIERS_STRATEGY:
            for col, strategy in strategy_dict.items():
                if col == CFG.TARGET_VARIABLE:
                    raise ValueError(
                        f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Outlier handling is not allowed on the target variable '{CFG.TARGET_VARIABLE}'. Please remove the target variable from the config file, in the 'deal_outliers_strategy' section."
                    )
                if strategy == "delete":
                    logging.info(
                        f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Deleting outliers for '{col}'"
                    )
                    df = df[df[col] < df[col].quantile(0.95)]
                elif strategy == "cap":
                    logging.info(
                        f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Capping outliers for '{col}'"
                    )
                    df[col] = df[col].clip(lower=df[col].quantile(0.05), upper=df[col].quantile(0.95))
                else:
                    raise ValueError(
                        f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Unknown outlier strategy '{strategy}' for column '{col}'. Please check the config file."
                    )
    else:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'deal_outliers' skipped because `CLEANING.OUTLIERS: {{}}`"
        )

    return df


def apply_specific_transformations(CFG: Config, df: pd.DataFrame) -> pd.DataFrame:
    if "specific_transformations.py" in os.listdir(CFG.DATA_FOLDER_PATH):
        if os.path.getsize(os.path.join(CFG.DATA_FOLDER_PATH, "specific_transformations.py")) == 0:
            logging.info(
                f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] 'specific_transformations.py' file exists but is empty, skipping specific transformations"
            )
            return df
        else:
            exec(open(f"{CFG.DATA_FOLDER_PATH}/specific_transformations.py").read())
            logging.info(
                f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Specific transformations from 'specific_transformations.py' applied"
            )
    else:
        logging.info(
            f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] No 'specific_transformations.py' file found, skipping specific transformations"
        )
    return df


def check_data_types(df: pd.DataFrame) -> None:
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()]\n{df.dtypes}")
    return
