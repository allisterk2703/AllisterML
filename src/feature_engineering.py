import inspect
import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import SelectKBest, f_classif

from src.config_loading import Config


def apply_feature_creation_and_transformations(CFG: Config, X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "feature_creation_and_transformations.py" in os.listdir(CFG.DATA_FOLDER_PATH):
        if os.path.getsize(os.path.join(CFG.DATA_FOLDER_PATH, "feature_creation_and_transformations.py")) == 0:
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] 'feature_creation_and_transformations.py' file exists but is empty, skipping feature creation and transformations")
            return X_train_scaled, X_test_scaled
        else:
            exec(open(f"{CFG.DATA_FOLDER_PATH}/feature_creation_and_transformations.py").read())
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Feature creation and transformations from 'feature_creation_and_transformations.py' applied")
    else:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] No 'feature_creation_and_transformations.py' file found, skipping feature creation and transformations")
    return X_train_scaled, X_test_scaled


def apply_dimensionality_reduction(
    CFG: Config,
    X_train_scaled: pd.DataFrame,
    X_test_scaled: pd.DataFrame,
    y_train_resampled: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CFG.DIMENSIONALITY_REDUCTION_STRATEGY:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Step 'apply_dimensionality_reduction' skipped because DIMENSIONALITY_REDUCTION_STRATEGY is None")
        return X_train_scaled, X_test_scaled

    else:
        logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Dimensionality reduction applied to dataset with {X_train_scaled.shape[1]} features.")
        if CFG.DIMENSIONALITY_REDUCTION_STRATEGY == "select_k_best":
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Applying SelectKBest with k={CFG.K_BEST_FEATURES}")
            # Choisir les 10 meilleures features
            select_k_best = SelectKBest(f_classif, k=CFG.K_BEST_FEATURES)
            X_train_selected = select_k_best.fit_transform(X_train_scaled, y_train_resampled)

            # Appliquer la même transformation sur les données de test
            X_test_selected = select_k_best.transform(X_test_scaled)

            # Obtenir les noms des features sélectionnées
            selected_features = X_train_scaled.columns[select_k_best.get_support()]
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Features sélectionnées : {list(selected_features)}")

            # Convertir les résultats en DataFrame pour garder les noms des
            # colonnes
            X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)

            return X_train_selected_df, X_test_selected_df

        elif CFG.DIMENSIONALITY_REDUCTION_STRATEGY == "pca":
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Applying PCA with n_components={CFG.PCA_COMPONENTS}")
            pca = PCA(n_components=CFG.PCA_COMPONENTS)
            X_train_pca = pca.fit_transform(X_train_scaled)
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {pca.n_components_} components were kept")
            X_test_pca = pca.transform(X_test_scaled)

            # Convertir les résultats en DataFrame pour garder les colonnes PCA
            # Noms des composantes principales
            pca_columns = [f"PC{i + 1}" for i in range(X_train_pca.shape[1])]
            X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns)
            X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns)

            return X_train_pca_df, X_test_pca_df

        elif CFG.DIMENSIONALITY_REDUCTION_STRATEGY == "tsvd":
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Applying TruncatedSVD with n_components={CFG.TSVD_COMPONENTS}")
            tsvd = TruncatedSVD(n_components=CFG.TSVD_COMPONENTS)
            X_train_tsvd = tsvd.fit_transform(X_train_scaled)
            logging.info(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] {tsvd.n_components_} components were kept")
            X_test_tsvd = tsvd.transform(X_test_scaled)

            # Convertir les résultats en DataFrame pour garder les colonnes TSVD
            # Noms des composantes
            tsvd_columns = [f"Component{i + 1}" for i in range(X_train_tsvd.shape[1])]
            X_train_tsvd_df = pd.DataFrame(X_train_tsvd, columns=tsvd_columns)
            X_test_tsvd_df = pd.DataFrame(X_test_tsvd, columns=tsvd_columns)

            return X_train_tsvd_df, X_test_tsvd_df
        else:
            raise ValueError(f"[{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] DIMENSIONALITY_REDUCTION_STRATEGY must be 'select_k_best', 'pca', or 'tsvd', received: {CFG.DIMENSIONALITY_REDUCTION_STRATEGY}")
