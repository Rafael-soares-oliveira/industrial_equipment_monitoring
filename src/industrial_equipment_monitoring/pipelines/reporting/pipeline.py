from kedro.pipeline import Node, Pipeline

from .nodes import generate_plots, generate_report_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=generate_report_data,
                inputs=["final_model", "test_data", "params:model"],
                outputs=["predictions", "report_data"],
                name="report_data",
            ),
            Node(
                func=generate_plots,
                inputs=["report_data"],
                outputs=[
                    "f1_plot",
                    "accuracy_plot",
                    "precision_recall_plot",
                    "comparative_metrics_plot",
                    "confusion_matrix_plot",
                    "feature_importance_plot",
                    "average_precision_curve_plot",
                    "roc_curve_plot",
                ],
                name="plots",
            ),
        ]
    )
