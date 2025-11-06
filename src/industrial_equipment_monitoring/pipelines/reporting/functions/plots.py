import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from industrial_equipment_monitoring.utils.logging_config import setup_logger

logger = setup_logger("Reporting_Plots")


def setup_plot_style():
    """Configure plot styles"""
    plt.style.use("default")
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def create_threshold_analysis_plot(report_data: dict[str, Any]) -> Figure:
    """
    Generate plot to analyze optimal threshold

    Args:
        report_data (dict[str, Any]): Predictions and calculed threshold.

    Raises:
        Exception: When an unexpected error occurs.

    Returns:
        Figure: Plotted graph
    """
    logger.info("Gerando gráfico de análise da curva threshold")
    try:
        setup_plot_style()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        threshold_data = report_data["threshold_analysis"]
        optimal_threshold = report_data["optimal_threshold"]
        optimal_metrics = report_data["metrics_at_optimal"]
        default_metrics = report_data["metrics_at_default"]

        # Subplot 1: F1-Score vs Threshold
        ax1.plot(
            threshold_data["thresholds"],
            threshold_data["f1_scores"],
            color="#2E86AB",
            linewidth=3,
            label="F1-Score",
        )
        ax1.axvline(
            x=optimal_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Ótimo: {optimal_threshold:.3f}",
        )
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("F1-Score")
        ax1.set_title("F1-Score vs Threshold", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Precision and Recall vs Threshold
        ax2.plot(
            threshold_data["thresholds"],
            threshold_data["precision_scores"],
            color="green",
            linewidth=2,
            label="Precision",
        )
        ax2.plot(
            threshold_data["thresholds"],
            threshold_data["recall_scores"],
            color="orange",
            linewidth=2,
            label="Recall",
        )
        ax2.axvline(x=optimal_threshold, color="red", linestyle="--", linewidth=2)
        ax2.set_xlabel("Threshold")
        ax2.set_ylabel("Score")
        ax2.set_title("Precision & Recall vs Threshold", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Accuracy vs Threshold
        ax3.plot(
            threshold_data["thresholds"],
            threshold_data["accuracy_scores"],
            color="purple",
            linewidth=2,
            label="Acurácia",
        )
        ax3.axvline(x=optimal_threshold, color="red", linestyle="--", linewidth=2)
        ax3.set_xlabel("Threshold")
        ax3.set_ylabel("Acurácia")
        ax3.set_title("Acurácia vs Threshold", fontweight="bold")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Subplot 4: Compare metrics
        metrics_names = ["Acurácia", "Precision", "Recall", "F1-Score"]
        optimal_values = [
            optimal_metrics[m] for m in ["accuracy", "precision", "recall", "f1"]
        ]
        default_values = [
            default_metrics[m] for m in ["accuracy", "precision", "recall", "f1"]
        ]

        x = np.arange(len(metrics_names))
        width = 0.35

        ax4.bar(
            x - width / 2,
            optimal_values,
            width,
            label=f"Threshold {optimal_threshold:.3f}",
            color="lightgreen",
        )
        ax4.bar(
            x + width / 2,
            default_values,
            width,
            label="Threshold 0.500",
            color="lightcoral",
        )

        ax4.set_xlabel("Métricas")
        ax4.set_ylabel("Score")
        ax4.set_title("Comparação: Threshold Ótimo vs Default", fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_names)
        ax4.legend()

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e


def create_roc_curve_plot(report_data: dict[str, Any]) -> Figure:
    """
    Generate plot for ROC Curve.

    Args:
        report_data (dict[str, Any]): Predictions and calculed threshold.

    Raises:
        Exception: When an unexpected error occurs

    Returns:
        Figure: Plotted graph
    """
    logger.info("Gerando gráfico de análise da curva ROC")
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))

        roc_data = report_data["roc_curve"]
        y_true = report_data["y_true"]
        y_pred_proba = report_data["y_pred_proba"]

        auc_score = roc_auc_score(y_true, y_pred_proba)

        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            color="#2E86AB",
            linewidth=3,
            label=f"ROC Curve (AUC = {auc_score:.3f})",
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Classificador Aleatório",
        )

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Curva ROC", fontweight="bold", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e


def create_precision_recall_curve_plot(report_data: dict[str, Any]) -> Figure:
    """
    Generate plot to analyze precision-recall curve.

    Args:
        report_data (dict[str, Any]): Predictions and calculed threshold.

    Raises:
        Exception: When an unexpected error occurs

    Returns:
        Figure: _Ploted graph
    """
    logger.info("Gerando gráfico de análise da curva de precisão-recall")
    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))

        pr_data = report_data["pr_curve"]
        y_true = report_data["y_true"]
        y_pred_proba = report_data["y_pred_proba"]

        ap_score = average_precision_score(y_true, y_pred_proba)
        positive_ratio = np.mean(y_true)

        ax.plot(
            pr_data["recall"],
            pr_data["precision"],
            color="#A23B72",
            linewidth=3,
            label=f"PR Curve (AP = {ap_score:.3f})",
        )
        ax.axhline(
            y=positive_ratio,
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Baseline (Positive Ratio: {positive_ratio:.3f})",
        )

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Curva Precision-Recall", fontweight="bold", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e


def create_confusion_matrix_plot(report_data: dict[str, Any]) -> Figure:
    """
    Generate plot to analyze confusion matrix to default threshold and optimal threshold.

    Args:
        report_data (dict[str, Any]): Predictions and calculed threshold.

    Raises:
        Exception: When an unexpected error occurs

    Returns:
        Figure: _Ploted graph
    """
    logger.info("Gerando gráfico de análise da matriz de confusão")
    try:
        margin_prediction = 0.5
        setup_plot_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        y_true = report_data["y_true"]
        y_pred_proba = report_data["y_pred_proba"]
        optimal_threshold = report_data["optimal_threshold"]

        # Matrix for optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        cm_optimal = confusion_matrix(y_true, y_pred_optimal)

        # Matrix for threshold default
        y_pred_default = (y_pred_proba >= margin_prediction).astype(int)
        cm_default = confusion_matrix(y_true, y_pred_default)

        # Plot optimal matrix
        sns.heatmap(
            cm_optimal,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax1,
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax1.set_xlabel("Predição")
        ax1.set_ylabel("Valor Real")
        ax1.set_xticklabels(["Negativo", "Positivo"])
        ax1.set_yticklabels(["Negativo", "Positivo"])
        ax1.set_title(f"Threshold Ótimo\n({optimal_threshold:.3f})", fontweight="bold")

        # Plot matrix default
        sns.heatmap(
            cm_default,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax2,
            cbar=False,
            annot_kws={"size": 12, "weight": "bold"},
        )
        ax2.set_xlabel("Predição")
        ax2.set_ylabel("Valor Real")
        ax2.set_xticklabels(["Negativo", "Positivo"])
        ax2.set_yticklabels(["Negativo", "Positivo"])
        ax2.set_title("Threshold Default\n(0.500)", fontweight="bold")

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        raise e


def create_feature_importance_plot(report_data: dict[str, Any]) -> Figure:
    """
    Generate plot to analyze feature importance.

    Args:
        report_data (dict[str, Any]): Predictions and calculed threshold.

    Raises:
        Exception: When an unexpected error occurs

    Returns:
        Figure: _Ploted graph
    """
    logger.info("Gerando gráfico de importância das features")

    try:
        setup_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))

        importance_data = report_data["feature_importance"]
        features_names = np.array(list(importance_data.keys()))
        importances = np.array(list(importance_data.values()))

        # Order by importance
        indices = np.argsort(importances)[::-1]
        sorted_features = features_names[indices]
        sorted_importances = importances[indices]

        sns.barplot(
            x=sorted_importances,
            y=sorted_features,
            palette="viridis",
            hue=sorted_features,
            ax=ax,
        )
        ax.set_title("Importância das Features", fontweight="bold")
        ax.set_xlabel("Importância")
        ax.set_ylabel("Features")

        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de importância das features: {e}")
        raise e


def _fig_to_pil(fig: Figure) -> Image.Image:
    """
    Convert matplotlib.figure.Figure to PIL.Image.Image

    Args:
        fig (Figure): Image with the format matplotlib.figure.Figure

    Returns:
        Image.Image: Image with the format PIL.Image.Image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    pil_image = Image.open(buf)
    plt.close(fig)
    return pil_image


def model_plot(report_data: dict[str, Any]) -> dict[str, Image.Image]:
    """
    Create a dictionary with the metrics plots

    Args:
        report_data (dict[str, Any]): Predictions and calculed threshold.

    Returns:
        dict[str, Image.Image]: Dictionary with the graphs of the model
    """
    threshold = create_threshold_analysis_plot(report_data)
    threshold_pil = _fig_to_pil(threshold)
    confusion = create_confusion_matrix_plot(report_data)
    confusion_pil = _fig_to_pil(confusion)
    precision_recall = create_precision_recall_curve_plot(report_data)
    precision_recall_pil = _fig_to_pil(precision_recall)
    roc_curve = create_roc_curve_plot(report_data)
    roc_curve_pil = _fig_to_pil(roc_curve)
    feature_importance = create_feature_importance_plot(report_data)
    feature_importance_pil = _fig_to_pil(feature_importance)

    return {
        "threshold_plot": threshold_pil,
        "confusion_matrix_plot": confusion_pil,
        "precision_recall_plot": precision_recall_pil,
        "roc_curve_plot": roc_curve_pil,
        "feature_importance_plot": feature_importance_pil,
    }
