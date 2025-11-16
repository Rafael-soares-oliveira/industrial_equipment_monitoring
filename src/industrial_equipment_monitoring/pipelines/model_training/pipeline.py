from kedro.pipeline import Node, Pipeline

from .nodes import train_final_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_final_model,
                inputs=["train_data", "params:model"],
                outputs={
                    "model": "final_model",
                    "best_params": "best_params",
                    "optimization_history": "optimization_history_plot",
                    "param_importance": "param_importance_plot",
                    "contour": "contour_plot",
                    "parallel_coordinate": "parallel_coordinate_plot",
                    "slice": "slice_plot",
                    "intermediate_values": "intermediate_values_plot",
                },
                name="train_final_model",
            )
        ]
    )
