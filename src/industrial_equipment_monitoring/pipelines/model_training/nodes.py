import pandas as pd
import xgboost as xgb

from industrial_equipment_monitoring.pipelines.model_training.get_xgb_params import (
    get_xgb_params,
)
from industrial_equipment_monitoring.utils.logging_config import setup_logger

from .optimize import optimized_hyperparameters

logger = setup_logger("ModelTraining")


def train_final_model(
    train_data: pd.DataFrame, params: dict
) -> tuple[xgb.Booster, dict[str, str | int | float]]:
    # TODO: registrar docstring
    """_summary_

    Args:
        featured_data (pd.DataFrame): _description_
        params (dict): _description_

    Returns:
        tuple[xgb.Booster, dict[str, str | int | float]]: _description_
    """
    logger.info("Treinando modelo final com os melhores hiperparâmetros")

    xgb_params = get_xgb_params(params)

    if train_data.empty:
        logger.error("Dataset está vazio")
        raise ValueError("Dataset vazio")

    X_train: pd.DataFrame = train_data.drop(columns=[xgb_params["target_col"]])
    y_train: pd.Series = train_data[xgb_params["target_col"]]

    best_params: dict[str, str | int | float] = optimized_hyperparameters(
        X_train, y_train, xgb_params
    )
    best_params.update(
        {
            "objective": xgb_params["objective"],
            "eval_metric": xgb_params["eval_metric"],
            "tree_method": xgb_params["tree_method"],
            "device": xgb_params["device"],
        }
    )

    dtrain: xgb.DMatrix = xgb.DMatrix(
        X_train, label=y_train, feature_names=X_train.columns.tolist()
    )

    booster: xgb.Booster = xgb.train(
        best_params,
        dtrain,
        num_boost_round=xgb_params["num_boost_round"],
        evals=[(dtrain, "train")],
        verbose_eval=False,
    )

    logger.info("Modelo final treinado com sucesso")
    return booster, best_params
