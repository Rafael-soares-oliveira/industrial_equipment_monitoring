"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import generate_report_data, generate_plots


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
                    "confusion_matrix_plot",
                    "threshold_analysis_plot",
                    "precision_recall_curve_plot",
                    "roc_curve_plot",
                    "feature_importance_plot",
                ],
                name="plots",
            ),
            # Intemediate node to group all the inputs in a dictionary. Used to generate report
            Node(
                func=lambda dataset_info,
                best_params,
                report_data,
                confusion_matrix_plot,
                threshold_analysis_plot,
                precision_recall_curve_plot,
                roc_curve_plot,
                feature_importance_plot,
                params: {
                    "dataset_info": dataset_info,
                    "best_params": best_params,
                    "report_data": report_data,
                    "confusion_matrix_plot": confusion_matrix_plot,
                    "threshold_analysis_plot": threshold_analysis_plot,
                    "precision_recall_curve_plot": precision_recall_curve_plot,
                    "roc_curve_plot": roc_curve_plot,
                    "feature_importance_plot": feature_importance_plot,
                    "params": params,
                },
                inputs=[
                    "dataset_info",
                    "best_params",
                    "report_data",
                    "confusion_matrix_plot",
                    "threshold_analysis_plot",
                    "precision_recall_curve_plot",
                    "roc_curve_plot",
                    "feature_importance_plot",
                    "params:model",
                ],
                outputs="report_inputs_dict",
                name="prepare_report_inputs",
            ),
        ]
    )
