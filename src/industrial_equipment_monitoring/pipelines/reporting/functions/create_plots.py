from typing import Any

from plotly import graph_objects as go

from industrial_equipment_monitoring.utils.logging_config import setup_logger

from .plots_functions.confusion_matrix import create_confusion_matrix_plot
from .plots_functions.feature_importance import create_feature_importance_plot
from .plots_functions.precision_recall import create_average_precision_curve_plot
from .plots_functions.roc_curve import create_roc_curve_plot
from .plots_functions.threshold_analysis import ThresholdAnalysis

logger = setup_logger("Reporting")


def model_plot(report_data: dict[str, Any]) -> dict[str, go.Figure]:
    logger.info("Gerando Gráficos")
    try:
        logger.info("Análise Threshold")
        threshold_analysis = ThresholdAnalysis(report_data)
        logger.info("F1 Score")
        f1: go.Figure = threshold_analysis.create_f1_score_plot()
        logger.info("Accuracy")
        accuracy: go.Figure = threshold_analysis.create_accuracy_plot()
        logger.info("Precision-Recall")
        precision_recall: go.Figure = threshold_analysis.create_precision_recall_plot()
        logger.info("Comparar métricas")
        compare_metrics: go.Figure = threshold_analysis.create_compare_metrics()

        logger.info("Matriz Confusão")
        confusion_matrix: go.Figure = create_confusion_matrix_plot(report_data)
        logger.info("Importância de Features")
        feature_importance: go.Figure = create_feature_importance_plot(report_data)
        logger.info("Precisão Média")
        average_precision: go.Figure = create_average_precision_curve_plot(report_data)
        logger.info("Curva ROC")
        roc_curve: go.Figure = create_roc_curve_plot(report_data)

        logger.info("Gráficos gerados")

        return {
            "f1": f1,
            "accuracy": accuracy,
            "precision_recall": precision_recall,
            "compare_metrics": compare_metrics,
            "confusion_matrix": confusion_matrix,
            "feature_importance": feature_importance,
            "average_precision": average_precision,
            "roc_curve": roc_curve,
        }

    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e
