from kedro.pipeline import Node, Pipeline

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
