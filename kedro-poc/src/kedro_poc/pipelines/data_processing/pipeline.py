from kedro.pipeline import Pipeline, node, pipeline

from .nodes import process_demo, process_riders, process_perm, process_term, process_ntl, process_di, process_fa, process_va, create_model_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=process_demo,
                inputs=["demographics", "zipincome"],
                outputs="client",
                name="process_demographics_node",
            ),
            node(
                func=process_riders,
                inputs="riders",
                outputs="rider_data",
                name="process_riders_node",
            ),
            node(
                func=process_perm,
                inputs=["perm", "rider_data"],
                outputs="permdata",
                name="process_perm_node",
            ),
            node(
                func=process_term,
                inputs=["term", "rider_data"],
                outputs="termdata",
                name="process_term_node",
            ),
            node(
                func=process_ntl,
                inputs=["ntl", "rider_data"],
                outputs="ntldata",
                name="process_ntl_node",
            ),
            node(
                func=process_di,
                inputs=["di", "rider_data"],
                outputs="didata",
                name="process_di_node",
            ),
            node(
                func=process_fa,
                inputs="fa",
                outputs="fadata",
                name="process_fa_node",
            ),
            node(
                func=process_va,
                inputs="va",
                outputs="vadata",
                name="process_va_node",
            ),
            node(
                func=create_model_data,
                inputs=["client", "fp_client", "permdata", "termdata", "ntldata", 'didata', "fadata", "vadata", "params:data_options"],
                outputs="model_data",
                name="create_model_data_node"
            ),
        ]
    )