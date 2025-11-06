from typing import Any

from pandas import DataFrame
from PIL.Image import Image
from xgboost import Booster

from industrial_equipment_monitoring.utils.logging_config import setup_logger

from .functions.plots import model_plot
from .functions.prediction import make_predictions

logger = setup_logger("Report")


def generate_report_data(
    model: Booster, test_data: DataFrame, params: dict
) -> tuple[DataFrame, dict]:
    """
    Pipeline node to obtain predictions and model metrics

    Args:
        model (Booster): Trained Model
        test_data (DataFrame): Test DataFrame
        params (dict): Parameters from parameters.yml

    Returns:
        tuple[DataFrame, dict]: Predictions and model metrics
    """
    predictions: DataFrame
    report_data: dict
    predictions, report_data = make_predictions(model, test_data, params)
    return predictions, report_data


def generate_plots(report_data: dict[str, Any]) -> list[Image]:
    """
    Pipeline node to obtain the model graphs.

    Args:
        report_data (dict[str, Any]): Metrics of the trained model.

    Raises:
        Exception: Unexpected error.

    Returns:
        list[Image]: List with model graphs
    """
    logger.info("Gerando gráficos")
    try:
        plots: dict[str, Image] = model_plot(report_data)

        logger.info("Gráficos gerados com sucesso.")

        return [
            plots["confusion_matrix_plot"],
            plots["threshold_plot"],
            plots["precision_recall_plot"],
            plots["roc_curve_plot"],
            plots["feature_importance_plot"],
        ]
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e
