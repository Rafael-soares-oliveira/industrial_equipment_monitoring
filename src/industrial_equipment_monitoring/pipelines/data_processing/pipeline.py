from kedro.pipeline import Node, Pipeline

from .nodes import load_clean_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=load_clean_data,
                inputs=["raw_data", "params:model"],
                outputs=["dataset_info", "cleaned_data"],
                name="load_and_clean_data",
            ),
            Node(
                func=split_data,
                inputs=["cleaned_data", "params:model"],
                outputs=["train_data", "test_data"],
                name="train_test_data",
            ),
        ]
    )
