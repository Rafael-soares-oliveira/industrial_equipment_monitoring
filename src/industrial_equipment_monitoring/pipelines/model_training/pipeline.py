"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import train_final_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=train_final_model,
                inputs=["train_data", "params:model"],
                outputs=["final_model", "best_params"],
                name="train_final_model",
            )
        ]
    )
