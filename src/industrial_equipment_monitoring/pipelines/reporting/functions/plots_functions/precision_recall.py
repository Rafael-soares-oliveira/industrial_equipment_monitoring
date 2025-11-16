from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import average_precision_score


def create_average_precision_curve_plot(report_data: dict[str, Any]) -> go.Figure:
    """
    Generate plot to analyze average precision curve

    Args:
        report_data (dict[str, Any]): Predictions and calculated threshold

    Returns:
        go.Figure: Plotted graph
    """

    pr_data = report_data["pr_curve"]
    y_true = report_data["y_true"]
    y_pred_proba = report_data["y_pred_proba"]

    ap_score = average_precision_score(y_true, y_pred_proba)
    positive_ratio = np.mean(y_true)

    fig = go.Figure()

    # Precision-Recall Curve
    fig.add_trace(
        go.Scatter(
            x=pr_data["recall"],
            y=pr_data["precision"],
            mode="lines",
            name=f"PR Curve (AP = {ap_score:.3f})",
            line=dict(color="#A23B72", width=3),
            fill="tozeroy",
            fillcolor="rgba(162, 59, 114, 0.1)",
            hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}",
        )
    )

    # Baseline (positive ratio)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[
                positive_ratio,
                positive_ratio,
            ],
            mode="lines",
            name=f"Baseline - Positive Rate: {positive_ratio:.2f}",
            line=dict(color="gray", dash="dash", width=2),
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(text="Precision-Recall", font=dict(size=18, weight="bold"), x=0.5),
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.15,
            xanchor="right",
            x=0.95,
        ),
    )

    fig.update_xaxes(range=[-0.05, 1.05])
    fig.update_yaxes(range=[-0.05, 1.05])

    return fig
