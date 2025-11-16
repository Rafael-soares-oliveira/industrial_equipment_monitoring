from typing import Any

from pandas import DataFrame
from plotly.graph_objects import Figure
from xgboost import Booster

from industrial_equipment_monitoring.utils.logging_config import setup_logger

from .functions.create_plots import model_plot
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


def generate_plots(report_data: dict[str, Any]) -> list[Figure]:
    """
    Pipeline node to obtain the model graphs.

    Args:
        report_data (dict[str, Any]): Metrics of the trained model.

    Raises:
        Exception: Unexpected error.

    Returns:
        list[Figure]: List with model graphs
    """

    plots: dict[str, Figure] = model_plot(report_data)

    return [
        plots["f1"],
        plots["accuracy"],
        plots["precision_recall"],
        plots["compare_metrics"],
        plots["confusion_matrix"],
        plots["feature_importance"],
        plots["average_precision"],
        plots["roc_curve"],
    ]
