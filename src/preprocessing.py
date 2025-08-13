import inspect
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config_loading import Config


def get_X_y(CFG: Config, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(CFG.TARGET_VARIABLE, axis=1)
    y = df[CFG.TARGET_VARIABLE]
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {X.shape}, {y.shape}")
    return X, y


def plot_target_variable(CFG: Config, df: pd.DataFrame) -> None:
    if CFG.PROBLEM_TYPE == "regression":
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()]\n{df[CFG.TARGET_VARIABLE].describe()}")

        plt.figure(figsize=(6, 4))
        sns.histplot(df[CFG.TARGET_VARIABLE], bins=5, kde=True)
        plt.xlabel("Target variable")
        plt.ylabel("Count")
        plt.title("Distribution of the target variable")
        plt.show()
    elif CFG.PROBLEM_TYPE == "classification":
        counts = df[CFG.TARGET_VARIABLE].value_counts()
        proportions = df[CFG.TARGET_VARIABLE].value_counts(normalize=True).round(2)
        summary_df = pd.DataFrame({"Count": counts, "Proportion": proportions}).reset_index().rename(columns={"index": CFG.TARGET_VARIABLE})
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()]\n{summary_df}")

        plt.figure(figsize=(6, 4))
        sns.countplot(
            data=df,
            x=CFG.TARGET_VARIABLE,
            hue=CFG.TARGET_VARIABLE,
            order=summary_df[CFG.TARGET_VARIABLE],
        )
        if CFG.TARGET_VARIABLE != CFG.TARGET_VARIABLE:  # inutile ici car toujours égal
            plt.legend(title="Target variable")
        plt.xlabel("Target variable")
        plt.ylabel("Count")
        plt.title("Distribution of the target variable")
        plt.show()
    return


def encode_target_variable(CFG: Config, y: pd.Series) -> pd.Series:
    if CFG.PROBLEM_TYPE == "regression":
        try:
            y = y.astype(float)
        except Exception as e:
            logging.error(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Error encoding target variable: {e}. Please check that the correct CFG.PROBLEM_TYPE is set in the config file.")
            raise e

    elif CFG.PROBLEM_TYPE == "classification":
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Target unique values: {y.unique()}")

        if CFG.TARGET_VARIABLE_ORDER:
            y_cat = pd.Categorical(y, categories=CFG.TARGET_VARIABLE_ORDER, ordered=True)
            if (y_cat.codes == -1).any():
                missing = sorted(set(y.unique()) - set(CFG.TARGET_VARIABLE_ORDER))
                raise ValueError(f"Classes manquantes dans ORDER: {missing}")
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Target variable encoded with order: {CFG.TARGET_VARIABLE_ORDER}")
            return y_cat.codes

        if y.dtype == "object":
            encoder_target = LabelEncoder()
            y = encoder_target.fit_transform(y)
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Target variable encoded without order: {y}")
        else:
            if 0 not in y.unique():  # xgboost, catboost, lightgbm attendent 0-based
                encoder_target = LabelEncoder()
                y = encoder_target.fit_transform(y)

    return y


def split_data(
    CFG: Config,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if CFG.PROBLEM_TYPE == "regression":
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    elif CFG.PROBLEM_TYPE == "classification":
        try:
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Splitting data into train and test sets (`stratify=y`)")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        except Exception as e:
            logging.error(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Error splitting data: {e}. Please check that the correct CFG.PROBLEM_TYPE is set in the config file.")
            raise e
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Train set: {X_train.shape} {y_train.shape}")
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Test set: {X_test.shape} {y_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_onehotencoder(CFG: Config, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    # Colonnes catégorielles détectées automatiquement
    categorical_columns = list(X_train.select_dtypes(include=["object", "category"]).columns)

    # Retirer celles qui sont déjà prévues pour un encodage ordinal
    onehot_columns = sorted(list(set(categorical_columns) - set(CFG.ORDINALENCODER_COLUMNS.keys())))

    if not onehot_columns:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'apply_onehotencoder' skipped: no categorical columns to one-hot encode.")
        return X_train.copy(), X_test.copy(), []

    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Applying OneHotEncoder to columns: {onehot_columns}")

    try:
        onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)
        onehotencoder.fit(X_train[onehot_columns])

        X_train_encoded = onehotencoder.transform(X_train[onehot_columns])
        X_test_encoded = onehotencoder.transform(X_test[onehot_columns])

        new_columns = list(onehotencoder.get_feature_names_out(onehot_columns))
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] OneHotEncoded new columns: {new_columns}")

        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=new_columns, index=X_train.index)
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=new_columns, index=X_test.index)

        X_train_final = pd.concat([X_train.drop(columns=onehot_columns), X_train_encoded_df], axis=1)
        X_test_final = pd.concat([X_test.drop(columns=onehot_columns), X_test_encoded_df], axis=1)

    except Exception as e:
        logging.error(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Error applying OneHotEncoder: {e}. Check data types/NaN in {onehot_columns}.")
        raise e

    return X_train_final, X_test_final, new_columns


def apply_ordinalencoder(
    CFG: Config,
    X_train_final: pd.DataFrame,
    X_test_final: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if CFG.ORDINALENCODER_COLUMNS:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Applying ordinalencoder to {list(CFG.ORDINALENCODER_COLUMNS.keys())}")

        col_names = []
        col_orders = []

        for col, order in CFG.ORDINALENCODER_COLUMNS.items():
            col_names.append(col)
            col_orders.append(list(order))

        enc = OrdinalEncoder(categories=col_orders, handle_unknown="use_encoded_value", unknown_value=-1)

        X_train_final[col_names] = enc.fit_transform(X_train_final[col_names])
        X_test_final[col_names] = enc.transform(X_test_final[col_names])
    else:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step skipped: apply_ordinalencoder")

    return X_train_final, X_test_final


def resample_data(
    CFG: Config,
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    CATEGORICAL_COLS: list[str] | None = None,
    show_counts: bool = True,
):
    problem = (CFG.PROBLEM_TYPE or "").strip().lower()
    if problem not in {"regression", "classification"}:
        raise ValueError(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] PROBLEM_TYPE doit être 'regression' ou 'classification', reçu: {CFG.PROBLEM_TYPE}")

    # Cas régression ou resampling non demandé
    if problem == "regression":
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] No resampling applied because PROBLEM_TYPE='regression'.")
        return X_train_final.copy(), y_train.copy()
    if problem == "classification" and not CFG.TARGET_VARIABLE_IMBALANCED_STRATEGY:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] No resampling applied")
        return X_train_final.copy(), y_train.copy()

    # Classification avec resampling
    strategy = (CFG.TARGET_VARIABLE_IMBALANCED_STRATEGY or "").strip().lower()
    if strategy not in {"smote", "smotenc", "random_over", "random_under"}:
        raise ValueError(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Unknown strategy: {CFG.TARGET_VARIABLE_IMBALANCED_STRATEGY}")

    if show_counts:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Before resampling : {Counter(y_train)}")

    if strategy == "smote":
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train_final, y_train)
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] SMOTE applied")

    elif strategy == "smotenc":
        if not CATEGORICAL_COLS:
            raise ValueError("SMOTENC requires a non-empty list of categorical columns (CATEGORICAL_COLS).")
        # S'assurer que les colonnes catégorielles existent
        missing = [c for c in CATEGORICAL_COLS if c not in X_train_final.columns]
        if missing:
            raise KeyError(f"Columns not found for SMOTENC: {missing}")

        # Convertir en 'category' et obtenir les indices
        X_tmp = X_train_final.copy()
        for col in CATEGORICAL_COLS:
            X_tmp[col] = X_tmp[col].astype("category")

        categorical_indices = [X_tmp.columns.get_loc(col) for col in CATEGORICAL_COLS]
        smotenc = SMOTENC(categorical_features=categorical_indices, random_state=42)
        X_res, y_res = smotenc.fit_resample(X_tmp, y_train)
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] SMOTENC applied")

    elif strategy == "random_over":
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train_final, y_train)
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] RandomOverSampler applied")

    elif strategy == "random_under":
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X_train_final, y_train)
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] RandomUnderSampler applied")

    if show_counts:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] After resampling  : {Counter(y_res)}")

    return X_res, y_res


def scale_data(
    X_train_resampled: pd.DataFrame,
    X_test_final: pd.DataFrame,
    onehot_encoded_new_columns: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        columns_to_scale = [c for c in X_train_resampled.columns if c not in onehot_encoded_new_columns]
        scaler = StandardScaler()
        X_train_resampled[columns_to_scale] = scaler.fit_transform(X_train_resampled[columns_to_scale])
        X_test_final[columns_to_scale] = scaler.transform(X_test_final[columns_to_scale])
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Scaled columns: {columns_to_scale}")
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Unscaled columns: {onehot_encoded_new_columns}")
        return X_train_resampled, X_test_final
    except Exception as e:
        logging.error(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Error scaling data: {e}. Please check the 'onehotencoder_boolean' parameter in the config file.")
        raise e


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    n_cols = df.shape[1]
    if n_cols < 20:
        annot = True
    else:
        annot = False
    size = 6 + (n_cols // 5)

    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(size, size))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=annot, fmt=".2f")
    plt.title(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Correlation Heatmap")
    plt.show()
    return


def show_most_correlated_features(CFG: Config, df: pd.DataFrame, n: int = 10) -> None:
    if CFG.TARGET_VARIABLE not in df.columns:
        raise ValueError(f"La colonne '{CFG.TARGET_VARIABLE}' n'existe pas dans le DataFrame.")

    corr = df.corr(numeric_only=True)[CFG.TARGET_VARIABLE].drop(labels=[CFG.TARGET_VARIABLE])
    top_features = corr.abs().sort_values(ascending=False).head(n).index

    log_lines = []
    for i, feat in enumerate(top_features, start=1):
        log_lines.append(f"{i}) {feat} : {corr[feat]:.2f}")

    formatted = "\n".join(log_lines)
    logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Top {n} most correlated features with '{CFG.TARGET_VARIABLE}':\n{formatted}")
    return
