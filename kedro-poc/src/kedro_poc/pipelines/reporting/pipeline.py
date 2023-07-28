from kedro.pipeline import Pipeline, node, pipeline
from .nodes import age_plot_px, age_plot_go

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=age_plot_px,
                inputs="model_data",
                outputs="age_plot_px"
            ),
            node(
                func=age_plot_go,
                inputs="model_data",
                outputs="age_plot_go"
            ),
        ]
    )
