from typing import Any

import numpy as np
import plotly.graph_objects as go


def create_feature_importance_plot(report_data: dict[str, Any]) -> go.Figure:
    """
    Generate plot to analyze feature importance

    Args:
        report_data (dict[str, Any]): Predictions and calculated threshold

    Returns:
        go.Figure: Plotted graph
    """

    importance_data: dict = report_data["feature_importance"]
    features_names = np.array(list(importance_data.keys()))
    importances = np.array(list(importance_data.values()))

    # Order by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = features_names[indices]
    sorted_importances = importances[indices]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=sorted_features,
            x=sorted_importances,
            orientation="h",
            marker=dict(
                color=sorted_importances,
                colorscale="Spectral",
                showscale=False,
                colorbar=dict(title="Importance"),
            ),
            textposition="inside",
            text=[f"{v:.2f}" for v in sorted_importances],
            textfont=dict(size=14, color="white"),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Features Importance", font=dict(size=18, weight="bold"), x=0.5
        ),
        xaxis_title="Importance",
        yaxis_title="Features",
        height=600,
        template="plotly_dark",
        showlegend=False,
    )

    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False, title_text="")

    return fig
