"""
This is a boilerplate pipeline 'reporting'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import generate_report_data, generate_plots, generate_report


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
            # Nó intermediário para agrupar todos os inputs em um dicionário
            Node(
                func=lambda logo,
                dataset_info,
                best_params,
                report_data,
                confusion_matrix_plot,
                threshold_analysis_plot,
                precision_recall_curve_plot,
                roc_curve_plot,
                feature_importance_plot,
                params: {
                    "logo": logo,
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
                    "logo",
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
            # Nó principal do relatório
            Node(
                func=generate_report,
                inputs=["report_inputs_dict"],
                outputs="temp",
                name="report",
            ),
        ]
    )
