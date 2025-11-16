from typing import Any

import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score


def create_roc_curve_plot(report_data: dict[str, Any]) -> go.Figure:
    """
    Generate plot for ROC Curve

    Args:
        report_data (dict[str, Any]): Predictions and calculated threshold

    Returns:
        go.Figure: Plotted graph
    """

    roc_data = report_data["roc_curve"]
    y_true = report_data["y_true"]
    y_pred_proba = report_data["y_pred_proba"]

    auc_score = roc_auc_score(y_true, y_pred_proba)

    fig = go.Figure()

    # ROC Curve
    fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f"ROC Curve (AUC = {auc_score:.3f})",
            line=dict(color="#28A8D7", width=3),
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.1)",
            hovertemplate="False Positive: %{x}<br>True Positive: %{y:.3f}",
        )
    )

    # Random classifier line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", dash="dash", width=2),
            opacity=0.7,
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(text="ROC Curve", font=dict(size=18, weight="bold"), x=0.5),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=800,
        legend=dict(yanchor="top", y=1.15, xanchor="right", x=0.95),
    )

    fig.update_xaxes(range=[-0.05, 1.05])
    fig.update_yaxes(range=[-0.05, 1.05])

    return fig
