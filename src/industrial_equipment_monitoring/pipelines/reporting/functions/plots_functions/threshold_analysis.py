from collections.abc import KeysView
from typing import Any

import numpy as np
from plotly import graph_objects as go

from industrial_equipment_monitoring.utils.logging_config import setup_logger

logger = setup_logger("Reporting")


class ThresholdAnalysis:
    colors: dict = {
        "f1": "#2E86AB",
        "precision": "#2CA02C",
        "recall": "#FF7F0E",
        "accuracy": "#6A4C93",
        "optimal_bar": "#31C53B",
        "default_bar": "#CD1322",
        "threshold_line": "crimson",
        "grid": "#e8e3e3",
        "background": "#ffffff",
        "text": "#333333",
    }

    my_template: dict = {
        "layout": {
            "paper_bgcolor": colors["background"],
            "plot_bgcolor": colors["background"],
            "font": {"family": "Arial", "size": 12, "color": colors["text"]},
            "xaxis": {
                "showgrid": True,
                "gridcolor": colors["grid"],
                "gridwidth": 0.5,
                "linecolor": colors["grid"],
                "linewidth": 0.5,
            },
            "yaxis": {
                "showgrid": True,
                "gridcolor": colors["grid"],
                "gridwidth": 0.5,
                "linecolor": colors["grid"],
                "linewidth": 0.5,
            },
        }
    }

    def __init__(self, report_data: dict[str, Any]):
        self.threshold_data: dict = report_data["threshold_analysis"]
        self.optimal_threshold: np.float64 = report_data["optimal_threshold"]
        self.optimal_metrics: dict = report_data["metrics_at_optimal"]
        self.default_metrics: dict = report_data["metrics_at_default"]
        self.threshold_method: str = report_data["threshold_method"]
        metrics_: KeysView[str] = report_data["metrics_at_default"].keys()
        self.metrics: dict[str, str] = {
            m: m.replace("_", " ").title() for m in metrics_
        }

    def create_f1_score_plot(self) -> go.Figure:
        """
        Generate F1-Score vs Threshold

        Returns:
            go.Figure: Plotted graph
        """
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.threshold_data["thresholds"],
                y=self.threshold_data["f1_scores"],
                mode="lines",
                name="F1-Score",
                line=dict(color=self.colors["f1"], width=1),
                hovertemplate="Threshold: %{x:.3f}<br>F1: %{y:.3f}",
                showlegend=False,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[self.optimal_threshold, self.optimal_threshold],
                y=[
                    min(self.threshold_data["f1_scores"]) * 0.98,
                    max(self.threshold_data["f1_scores"]) * 1.02,
                ],
                mode="lines",
                name=f"Optimal Threshold: {self.optimal_threshold}",
                line=dict(color=self.colors["threshold_line"], dash="dash", width=2),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"F1-Score - {self.threshold_method}",
                font=dict(size=18, weight="bold"),
                x=0.5,
            ),
            xaxis_title="Threshold",
            yaxis_title="F1-Score",
            height=600,
            width=800,
            showlegend=True,
            template=self.my_template,
            legend=dict(orientation="v", yanchor="top", y=1.15, xanchor="right", x=1.1),
        )

        fig.update_xaxes(range=[0.05, 0.95])

        return fig

    def create_precision_recall_plot(self) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.threshold_data["thresholds"],
                y=self.threshold_data["precision_scores"],
                mode="lines",
                name="Precision",
                line=dict(color=self.colors["precision"], width=2),
                hovertemplate="Threshold: %{x:.3f}<br>Precision: %{y:.3f}",
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=self.threshold_data["thresholds"],
                y=self.threshold_data["recall_scores"],
                mode="lines",
                name="Recall",
                line=dict(color=self.colors["recall"], width=2),
                hovertemplate="Threshold: %{x:.3f}<br>Recall: %{y:.3f}",
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[self.optimal_threshold, self.optimal_threshold],
                y=[
                    min(self.threshold_data["precision_scores"]) * 0.98,
                    max(self.threshold_data["precision_scores"]) * 1.02,
                ],
                mode="lines",
                name=f"Optimal Threshold: {self.optimal_threshold}",
                line=dict(color=self.colors["threshold_line"], dash="dash", width=2),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Precision vs Recall - {self.threshold_method}",
                font=dict(size=18, weight="bold"),
                x=0.5,
            ),
            xaxis_title="Threshold",
            yaxis_title="Precision - Recall",
            height=600,
            width=800,
            showlegend=True,
            template=self.my_template,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.25,
                xanchor="right",
                x=1.1,
            ),
        )

        fig.update_xaxes(range=[0.05, 0.95])

        return fig

    def create_accuracy_plot(self) -> go.Figure:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.threshold_data["thresholds"],
                y=self.threshold_data["accuracy_scores"],
                mode="lines",
                name="Accuracy",
                line=dict(color=self.colors["accuracy"], width=2),
                hovertemplate="Threshold: %{x:.3f}<br>Accuracy: %{y:.3f}",
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=[self.optimal_threshold, self.optimal_threshold],
                y=[
                    min(self.threshold_data["accuracy_scores"]) * 0.98,
                    max(self.threshold_data["accuracy_scores"]) * 1.02,
                ],
                mode="lines",
                name=f"Optimal Threshold: {self.optimal_threshold}",
                line=dict(color=self.colors["threshold_line"], dash="dash", width=2),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Accuracy - {self.threshold_method}",
                font=dict(size=18, weight="bold"),
                x=0.5,
            ),
            xaxis_title="Threshold",
            yaxis_title="Accuracy",
            height=600,
            width=800,
            showlegend=True,
            template=self.my_template,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.15,
                xanchor="right",
                x=1.1,
            ),
        )

        fig.update_xaxes(range=[0.05, 0.95])

        return fig

    def create_compare_metrics(self) -> go.Figure:
        fig = go.Figure()

        optimal_values: list = [self.optimal_metrics[m] for m in self.metrics.keys()]
        default_values: list = [self.default_metrics[m] for m in self.metrics.keys()]

        fig.add_trace(
            go.Bar(
                name=f"Threshold {self.optimal_threshold:.3f}",
                x=list(self.metrics.values()),
                y=optimal_values,
                marker_color=self.colors["optimal_bar"],
                offsetgroup=0,
                textposition="inside",
                text=[f"{v:.2f}" for v in optimal_values],
                textfont=dict(size=10, color="black"),
                hovertemplate="%{x}<br>Optimal: %{y:.3f}",
                showlegend=True,
            ),
        )

        fig.add_trace(
            go.Bar(
                name="Threshold 0.5",
                x=list(self.metrics.values()),
                y=default_values,
                marker_color=self.colors["default_bar"],
                offsetgroup=1,
                textposition="inside",
                text=[f"{v:.2f}" for v in default_values],
                textfont=dict(size=10, color="black"),
                hovertemplate="%{x}<br>Default: %{y:.3f}",
                showlegend=True,
            ),
        )

        fig.update_layout(
            title=dict(
                text=f"Metrics Comparative - {self.threshold_method}",
                font=dict(size=18, weight="bold"),
                x=0.5,
            ),
            xaxis_title="Metrics",
            yaxis_title="Score",
            height=600,
            width=800,
            showlegend=True,
            template=self.my_template,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.15,
                xanchor="right",
                x=0.95,
            ),
        )

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, title_text="")

        return fig
