from __future__ import annotations

import inspect
import logging
from pathlib import Path
from pprint import pprint
from typing import Dict
from typing import List
from typing import List as TList
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, conint, model_validator


def read_config(config_file_path: Path, print_config: bool = False) -> dict:
    with open(config_file_path, "r") as f:
        cfg = yaml.safe_load(f)
    if print_config:
        pprint(cfg)
    return cfg


# --------- Pydantic Schemas ---------


class TargetConfig(BaseModel):
    """
    Attributs :
        PROBLEM_TYPE : Doit être exactement "regression" ou "classification".
        CLASSIFICATION_TYPE : Facultatif. Si renseigné, doit être "binary" ou "multiclass".
                              Obligatoire si PROBLEM_TYPE = "classification".
                              Doit rester vide si PROBLEM_TYPE = "regression".
        NAME : Chaîne obligatoire contenant le nom exact de la variable cible
               tel qu’il apparaît dans le dataset original.
        ORDER : Facultatif. Liste de chaînes représentant l’ordre attendu
                des classes ou des valeurs cibles.

    Validation :
        - Si PROBLEM_TYPE = "classification" → CLASSIFICATION_TYPE doit être renseigné.
        - Si PROBLEM_TYPE = "regression" → CLASSIFICATION_TYPE doit être vide ou `null`.
    """

    model_config = {"extra": "forbid"}

    PROBLEM_TYPE: Literal["regression", "classification"]
    CLASSIFICATION_TYPE: Optional[Literal["binary", "multiclass"]] = None
    NAME: str = Field(...)
    ORDER: Optional[List[str]] = None

    @model_validator(mode="after")
    def _check_classification_type(self) -> "TargetConfig":
        if self.PROBLEM_TYPE == "classification" and self.CLASSIFICATION_TYPE is None:
            raise ValueError("TARGET.CLASSIFICATION_TYPE est obligatoire quand TARGET.PROBLEM_TYPE = 'classification'")
        if self.PROBLEM_TYPE == "regression" and self.CLASSIFICATION_TYPE is not None:
            logging.warning("TARGET.CLASSIFICATION_TYPE doit être commenté, vide ou `null` quand TARGET.PROBLEM_TYPE = 'regression'")
        return self


class ColumnsConfig(BaseModel):
    """
    Attributs :
        RENAME : Facultatif. Dictionnaire {ancien_nom: nouveau_nom} pour renommer les colonnes.
        DROP : Facultatif. Liste des noms de colonnes à supprimer avant les transformations.
        DROP_AFTER_TRANSFORMATIONS : Facultatif. Liste des noms de colonnes à supprimer après les transformations.
    """

    model_config = {"extra": "forbid"}

    RENAME: Optional[Dict[str, str]] = None
    DROP: Optional[List[str]] = None
    DROP_AFTER_TRANSFORMATIONS: Optional[List[str]] = None


class CleaningConfig(BaseModel):
    """
    Attributs :
        REPLACE_BY_NAN : Facultatif. Liste de valeurs à remplacer par NaN dans toutes les colonnes.
        REPLACE_MAP : Facultatif. Dictionnaire {ancienne_valeur: nouvelle_valeur} appliqué à tout le dataset.
        REPLACE_IN_SPECIFIC_COLUMNS : Facultatif. Dictionnaire {colonne: {ancienne_valeur: nouvelle_valeur}}
                                      pour remplacer des valeurs dans des colonnes précises.
        TYPE_CONVERSION : Facultatif. Dictionnaire {colonne: type} pour convertir le type de données d'une colonne.
        IMPUTATION : Facultatif. Dictionnaire {colonne: valeur} pour imputer les valeurs manquantes.
        OUTLIERS : Facultatif. Dictionnaire {colonne: stratégie} pour traiter les valeurs aberrantes.
    """

    model_config = {"extra": "forbid"}

    REPLACE_BY_NAN: Optional[List[str]] = None
    REPLACE_MAP: Optional[Dict[str, str]] = None
    REPLACE_IN_SPECIFIC_COLUMNS: Optional[Dict[str, Dict[str, Union[str, int, float]]]] = None
    TYPE_CONVERSION: Optional[Dict[str, str]] = None
    IMPUTATION: Optional[Dict[str, Union[str, float, int]]] = None
    OUTLIERS: Optional[Dict[str, str]] = None


class EncodingConfig(BaseModel):
    """
    Attributs :
        ORDINAL : Facultatif. Dictionnaire {colonne: [valeur1, valeur2, ...]}
                  définissant l'ordre des catégories pour un encodage ordinal.
    """

    model_config = {"extra": "forbid"}

    ORDINAL: Optional[Dict[str, List[str]]] = None


class ImbalanceConfig(BaseModel):
    """
    Attributs :
        STRATEGY : Facultatif. Méthode de rééquilibrage des classes,
                   parmi "smote", "smotenc", "random_over", "random_under".
                   Peut aussi être omis ou laissé vide.
    """

    model_config = {"extra": "forbid"}

    STRATEGY: Optional[Literal["smote", "smotenc", "random_over", "random_under"]] = None


class DimensionalityReductionConfig(BaseModel):
    """
    Attributs :
        STRATEGY : Facultatif. Méthode de réduction de dimensions, parmi
                   "select_k_best", "pca", "tsvd". Peut aussi être omis ou laissé vide.
        K_BEST_FEATURES : Facultatif. Nombre de meilleures variables à garder
                          si STRATEGY = "select_k_best".
        PCA_COMPONENTS : Facultatif. Nombre (int) ou proportion (float)
                         de composantes principales à garder si STRATEGY = "pca".
        TSVD_COMPONENTS : Facultatif. Nombre (int) ou proportion (float)
                          de composantes à garder si STRATEGY = "tsvd".

    Validation :
        - Un seul paramètre parmi K_BEST_FEATURES, PCA_COMPONENTS, TSVD_COMPONENTS
          peut être défini.
        - STRATEGY détermine quel paramètre doit obligatoirement être renseigné :
            * select_k_best → K_BEST_FEATURES requis
            * pca → PCA_COMPONENTS requis
            * tsvd → TSVD_COMPONENTS requis
    """

    model_config = {"extra": "forbid"}

    STRATEGY: Optional[Literal["select_k_best", "pca", "tsvd"]] = None
    K_BEST_FEATURES: Optional[int] = None
    PCA_COMPONENTS: Optional[Union[int, float]] = Field(default=None, ge=0, le=1, description="Si float : proportion entre 0 et 1. Si int : nombre de composantes.")
    TSVD_COMPONENTS: Optional[Union[int, float]] = Field(default=None, ge=0, le=1, description="Si float : proportion entre 0 et 1. Si int : nombre de composantes.")

    @model_validator(mode="after")
    def _check_params_exclusivity(self) -> "DimensionalityReductionConfig":
        if self.STRATEGY == "select_k_best" and self.K_BEST_FEATURES is None:
            raise ValueError("DIMENSIONALITY_REDUCTION.K_BEST_FEATURES est requis quand STRATEGY = 'select_k_best'")
        if self.STRATEGY == "pca" and self.PCA_COMPONENTS is None:
            raise ValueError("DIMENSIONALITY_REDUCTION.PCA_COMPONENTS est requis quand STRATEGY = 'pca'")
        if self.STRATEGY == "tsvd" and self.TSVD_COMPONENTS is None:
            raise ValueError("DIMENSIONALITY_REDUCTION.TSVD_COMPONENTS est requis quand STRATEGY = 'tsvd'")
        return self


class TrainingConfig(BaseModel):
    """
    Attributs :
        MODELS : Dictionnaire indiquant les modèles à entraîner,
                 structuré par type de problème (ex: "CLASSIFICATION", "REGRESSION"),
                 chaque clé de sous-dictionnaire représentant un modèle avec 1 = actif, 0 = inactif.
        GRIDSEARCH : Booléen. Indique si une recherche d’hyperparamètres (GridSearch) doit être effectuée.
        CROSS_VALIDATION : Dictionnaire des paramètres de validation croisée
                           (ex: {"CV": 5, "SHUFFLE": True}).
        SAVE_RESULTS_TO_CSV : Booléen. Sauvegarder ou non les résultats dans un fichier CSV.
        LOG_IN_MLFLOW : Booléen. Activer ou non le logging des expériences dans MLflow.
        PARAM_GRIDS : Dictionnaire contenant les grilles d’hyperparamètres,
                      structuré par type de problème, puis par modèle,
                      puis par nom de paramètre avec une liste de valeurs possibles.
    """

    model_config = {"extra": "forbid"}

    MODELS: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    GRIDSEARCH: bool = False
    CROSS_VALIDATION: bool = True
    N_FOLDS: conint(ge=2) = 5
    SAVE_RESULTS_TO_CSV: bool = True
    LOG_IN_MLFLOW: bool = False
    PARAM_GRIDS: Dict[str, Dict[str, Dict[str, TList[Optional[Union[int, float, str, bool]]]]]] = Field(default_factory=dict)


class FullConfig(BaseModel):
    """
    Attributs :
        TARGET : Configuration de la variable cible (voir TargetConfig).
        COLUMNS : Configuration des colonnes du dataset (voir ColumnsConfig).
        CLEANING : Configuration du nettoyage des données (voir CleaningConfig).
        ENCODING : Configuration de l’encodage des variables (voir EncodingConfig).
        IMBALANCE : Configuration du rééquilibrage des classes (voir ImbalanceConfig).
        DIMENSIONALITY_REDUCTION : Configuration de la réduction de dimensions (voir DimensionalityReductionConfig).
        TRAINING : Configuration de l’entraînement des modèles (voir TrainingConfig).

    Validation :
        - Si PROBLEM_TYPE = "regression" → STRATEGY de IMBALANCE est automatiquement remis à None
          pour éviter toute incohérence.
    """

    model_config = {"extra": "forbid"}

    TARGET: TargetConfig
    COLUMNS: ColumnsConfig = Field(default_factory=ColumnsConfig)
    CLEANING: CleaningConfig = Field(default_factory=CleaningConfig)
    ENCODING: EncodingConfig = Field(default_factory=EncodingConfig)
    IMBALANCE: ImbalanceConfig = Field(default_factory=ImbalanceConfig)
    DIMENSIONALITY_REDUCTION: DimensionalityReductionConfig = Field(default_factory=DimensionalityReductionConfig)
    TRAINING: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="after")
    def _regression_resets_imbalance(self) -> "FullConfig":
        if self.TARGET.PROBLEM_TYPE == "regression" and self.IMBALANCE.STRATEGY is not None:
            self.IMBALANCE.STRATEGY = None
        return self


# -----------------------------------------------


class Config:
    def __init__(self, dataset_name: Path, print_config: bool = False):
        self.DATASET_NAME = dataset_name
        self.DATA_FOLDER_PATH = Path("../data/") / self.DATASET_NAME
        self.CONFIG_FILE_PATH = self.DATA_FOLDER_PATH / "config.yaml"

        # Lecture du fichier de configuration
        cfg = read_config(self.CONFIG_FILE_PATH, print_config)

        # Validation Pydantic
        validated = FullConfig(**cfg)

        # Mapping des attributs
        self.PROBLEM_TYPE = validated.TARGET.PROBLEM_TYPE
        self.CLASSIFICATION_TYPE = validated.TARGET.CLASSIFICATION_TYPE
        self.TARGET_VARIABLE = validated.TARGET.NAME.lower().replace(" ", "_")
        self.TARGET_VARIABLE_ORDER = validated.TARGET.ORDER or []

        self.RENAME_COLUMNS = validated.COLUMNS.RENAME or {}
        self.UNNECESSARY_COLUMNS = validated.COLUMNS.DROP or []
        self.UNNECESSARY_COLUMNS_AFTER_TRANSFORMATIONS = validated.COLUMNS.DROP_AFTER_TRANSFORMATIONS or []

        self.ELEMENTS_TO_REPLACE_BY_NAN = validated.CLEANING.REPLACE_BY_NAN or []
        self.REPLACE_ELEMENTS_BY_SOMETHING = validated.CLEANING.REPLACE_MAP or {}
        self.REPLACE_IN_SPECIFIC_COLUMNS = validated.CLEANING.REPLACE_IN_SPECIFIC_COLUMNS or {}
        self.DATA_TYPE_CONVERSION = validated.CLEANING.TYPE_CONVERSION or {}
        self.MISSING_VALUES_IMPUTATION_STRATEGY = validated.CLEANING.IMPUTATION or {}
        self.DEAL_OUTLIERS_STRATEGY = validated.CLEANING.OUTLIERS or {}

        self.ORDINALENCODER_COLUMNS = validated.ENCODING.ORDINAL or {}

        self.TARGET_VARIABLE_IMBALANCED_STRATEGY = validated.IMBALANCE.STRATEGY

        self.DIMENSIONALITY_REDUCTION_STRATEGY = validated.DIMENSIONALITY_REDUCTION.STRATEGY
        self.K_BEST_FEATURES = validated.DIMENSIONALITY_REDUCTION.K_BEST_FEATURES
        self.PCA_COMPONENTS = validated.DIMENSIONALITY_REDUCTION.PCA_COMPONENTS
        self.TSVD_COMPONENTS = validated.DIMENSIONALITY_REDUCTION.TSVD_COMPONENTS

        self.TRAINING_MODELS = validated.TRAINING.MODELS
        self.TRAINING_GRIDSEARCH = validated.TRAINING.GRIDSEARCH
        self.TRAINING_CROSS_VALIDATION = validated.TRAINING.CROSS_VALIDATION
        self.TRAINING_CV_FOLDS = validated.TRAINING.N_FOLDS
        self.SAVE_RESULTS_TO_CSV = validated.TRAINING.SAVE_RESULTS_TO_CSV
        self.LOG_IN_MLFLOW = validated.TRAINING.LOG_IN_MLFLOW
        self.PARAM_GRIDS = validated.TRAINING.PARAM_GRIDS
        self.PARAM_GRIDS_REGRESSION = self.PARAM_GRIDS.get("REGRESSION", {})
        self.PARAM_GRIDS_CLASSIFICATION = self.PARAM_GRIDS.get("CLASSIFICATION", {})

        logging.info(f"[{self.__class__.__name__}.{Path(__file__).stem}.{inspect.currentframe().f_code.co_name}()] Config loaded from {self.CONFIG_FILE_PATH}")

    def print_all(self) -> None:
        pprint(vars(self))


if __name__ == "__main__":
    CFG = Config("diabete")
    CFG.print_all()
