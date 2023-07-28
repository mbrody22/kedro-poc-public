from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    pipeline_instance = pipeline(
        [
            node(
                func=split_data,
                inputs=["model_data", "params:model_options"],
                outputs=["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"],
                name="split_data_node",
            ),
            #node(
            #    func=tune_model,
            #    inputs=["X_train", "y_train", "X_val", "y_val"],
            #    outputs="hyperparameters",
            #    name="model_tuning_node",
            #),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "params:model_options"],
                outputs="booster",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["booster", "X_test", "y_test", "params:model_options"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
    ds_pipeline_1 = pipeline(
        pipe=pipeline_instance,
        inputs="model_data",
        namespace="model_version_1",
    )
    ds_pipeline_2 = pipeline(
        pipe=pipeline_instance,
        inputs="model_data",
        namespace="model_version_2",
    )

    return ds_pipeline_1 + ds_pipeline_2