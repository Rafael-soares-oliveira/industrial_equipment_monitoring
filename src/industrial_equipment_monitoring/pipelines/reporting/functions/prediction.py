from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from industrial_equipment_monitoring.utils.logging_config import setup_logger

logger = setup_logger("PredictionNode")


def find_optimal_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, method: str = "f1"
) -> tuple[float, float]:
    """
    Encontra o threshold ótimo baseado em diferentes métricas
    """
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold = 0.5
    best_score = 0

    if method == "f1":
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold

    elif method == "youden":
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds_roc[best_idx]
        best_score = youden_index[best_idx]

    elif method == "precision_recall":
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            score = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )
            if score > best_score:
                best_score = score
                best_threshold = threshold

    elif method == "recall":
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            score = recall_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold

    logger.info(
        f"Melhor threshold ({method}): {best_threshold:.4f} (score: {best_score:.4f})"
    )
    return best_threshold, float(best_score)


def calculate_metrics_at_threshold(
    y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float
) -> Mapping[str, float]:
    """
    Calcula todas as métricas para um threshold específico
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
    }


def calculate_threshold_analysis_data(
    y_true: np.ndarray, y_pred_proba: np.ndarray, num_thresholds: int = 50
) -> dict[str, np.ndarray]:
    """
    Calcula dados para o gráfico de análise de threshold
    """
    thresholds = np.linspace(0.1, 0.9, num_thresholds)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
        accuracy_scores.append(accuracy_score(y_true, y_pred))

    return {
        "thresholds": thresholds,
        "f1_scores": np.array(f1_scores),
        "precision_scores": np.array(precision_scores),
        "recall_scores": np.array(recall_scores),
        "accuracy_scores": np.array(accuracy_scores),
    }


def feature_importance(
    trained_model: xgb.Booster, importance_type: str = "gain"
) -> Mapping[str, float | list[float]]:
    logger.info(f"Extraindo importância das features usando tipo: {importance_type}")
    try:
        importance_raw = trained_model.get_score(importance_type=importance_type)
        if not importance_raw:
            logger.error("Nenhuma importância de feature foi encontrada")
            raise KeyError("Nenhuma importância de feature foi encontrada.")

        # Order by importance desc
        sorted_importance = dict(
            sorted(importance_raw.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_importance
    except Exception as e:
        logger.error(f"Erro ao extrair importância das features: {e}")
        raise e


def make_predictions(
    trained_model: xgb.Booster, test_data: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Node do Kedro para fazer predições e calcular métricas para relatório

    Returns:
        Tuple com:
        - DataFrame com predições (apenas colunas necessárias)
        - Dict com todas as métricas e dados para gráficos
    """
    logger.info("Iniciando geração de predições e cálculo de métricas")

    # Obter configurações de predição
    target_col = params["featured_data"]["target_column"]
    threshold_method = params["reporting"]["threshold_method"]
    importance_type = params["reporting"]["importance_type"]

    # Preparar dados para predição
    if target_col not in test_data.columns:
        logger.error(f"Coluna target '{target_col}' não encontrada nos dados de teste")
        raise ValueError(f"Target column '{target_col}' not found in test data")

    X_test = test_data.drop(columns=[target_col])
    y_true = test_data[target_col]

    logger.info(f"Dados de teste com target disponível. Shape: {X_test.shape}")
    logger.info(f"Distribuição do target: {y_true.value_counts().to_dict()}")

    # Criar DMatrix para predição
    try:
        dtest = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist())

        # Fazer predições
        y_pred_proba = trained_model.predict(dtest)

        # Encontrar threshold ótimo
        optimal_threshold, optimal_score = find_optimal_threshold(
            y_true.values, y_pred_proba, method=threshold_method
        )

        # Calcular métricas nos thresholds
        optimal_metrics = calculate_metrics_at_threshold(
            y_true.values, y_pred_proba, optimal_threshold
        )
        default_metrics = calculate_metrics_at_threshold(
            y_true.values, y_pred_proba, 0.5
        )

        # Calcular dados para gráficos
        threshold_data = calculate_threshold_analysis_data(y_true.values, y_pred_proba)

        # Calcular dados para curva ROC
        fpr, tpr, _ = roc_curve(y_true.values, y_pred_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_true.values, y_pred_proba
        )

        logger.info(f"Threshold ótimo encontrado: {optimal_threshold:.4f}")
        logger.info(f"F1-Score no threshold ótimo: {optimal_metrics['f1']:.4f}")

    except Exception as e:
        logger.error(f"Erro ao fazer predições: {e}")
        raise e

    # Criar DataFrame apenas com predições (sem features originais)
    results_df = pd.DataFrame(
        {
            "prediction_probability": y_pred_proba,
            "prediction_binary": (y_pred_proba >= optimal_threshold).astype(int),
            "actual_target": y_true.values,
        }
    )

    # Métricas e dados completos para relatório
    report_data = {
        # Métricas principais
        "optimal_threshold": optimal_threshold,
        "optimal_score": optimal_score,
        "threshold_method": threshold_method,
        "metrics_at_optimal": optimal_metrics,
        "metrics_at_default": default_metrics,
        # Dados para gráficos
        "threshold_analysis": threshold_data,
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "pr_curve": {"precision": precision_curve, "recall": recall_curve},
        # Dados para matriz de confusão
        "y_true": y_true.values,
        "y_pred_proba": y_pred_proba,
        # Estatísticas básicas
        "test_data_shape": test_data.shape,
        "target_distribution": y_true.value_counts().to_dict(),
        "feature_importance": feature_importance(trained_model, importance_type),
    }

    return results_df, report_data
