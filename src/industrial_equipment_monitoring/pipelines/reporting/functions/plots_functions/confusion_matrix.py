from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix


def create_confusion_matrix_plot(report_data: dict[str, Any]) -> go.Figure:
    margin_prediction = 0.5
    y_true = report_data["y_true"]
    y_pred_proba = report_data["y_pred_proba"]
    optimal_threshold = report_data["optimal_threshold"]

    # Matrix for optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_true, y_pred_optimal)

    # Matrix for threshold default
    y_pred_default = (y_pred_proba >= margin_prediction).astype(int)
    cm_default = confusion_matrix(y_true, y_pred_default)

    # Labels
    x_labels = ["Negative", "Positive"]
    y_labels = ["Negative", "Positive"]

    # Values annotations
    text_optimal = cm_optimal.astype(str)
    text_default = cm_default.astype(str)

    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Optimal Threshold ({optimal_threshold:.3f})",
            "Threshold Default (0.5)",
        ],
        horizontal_spacing=0.1,
    )

    # Optimal threshold confusion matrix
    fig.add_trace(
        go.Heatmap(
            z=cm_optimal,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            showscale=False,
            text=text_optimal,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            hovertemplate="True: %{y}<br>Prediction: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Default threshold confusion matrix
    fig.add_trace(
        go.Heatmap(
            z=cm_default,
            x=x_labels,
            y=y_labels,
            colorscale="Blues",
            showscale=False,
            text=text_default,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "black"},
            hovertemplate="True: %{y}<br>Prediction: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text="<b>Confusion Matrix - Thresholds Comparative</b>",
            x=0.5,
            font=dict(size=18),
        ),
        height=500,
        margin=dict(l=40, r=40),
    )

    fig.update_xaxes(title_text="Prediction", row=1, col=1)
    fig.update_yaxes(title_text="Original", row=1, col=1)
    fig.update_xaxes(title_text="Prediction", row=1, col=2)
    fig.update_yaxes(title_text="Original", row=1, col=2)

    return fig
