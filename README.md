# kedro-poc
This is a proof of concept Data Science pipeline using kedro. Kedro is based on the the idea of "nodes"
which are essentially wrappers for pythons functions that take inputs and return outputs. 
Nodes can than be chained together into pipelines which define the dependencies of each node and the order of their execution. 

## Environment setup

Create Python environment

    python3.9 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r src/requirements.txt

## Repo Structure

- **kedro-poc**
  - **conf**
    - **base**
      - **parameters**
         - **data_processing.yaml**: Config file that defines all parameters needed in the data processing pipeline (Ex: outcome column name)
         - **data_science.yaml**: Config file that defines all parameters needed in the data science pipeline (Ex: model hyperparameters)
         - **reporting.yaml**: Config file that defines all parameters needed in the reporting pipeline
      - **catalog.yaml**: Config file that defines all the datasets called and created by nodes within this project
  - **src**
    - **kedro_poc**
      - **pipelines**
        - **data_processing**
          - **nodes.py**: defines functions to perform data processing and feature engineering steps on the raw data
          - **pipeline.py**: configures the entire data processing pipeline by chaining together nodes and defining their inputs and outputs
        - **data_science**
          - **nodes.py**: defines functions to split the processed modeling data, train an XGBoost model, and evaluate it
          - **pipeline.py**: configures two versions of the model training pipeline by chaining together nodes and defining their inputs and outputs
        - **reporting**
          - **nodes.py**: defines functions to generate boxplots using two Kedro-compatible methods, Plotly Go and Plotly Express
          - **pipeline.py**: defines the reporting pipeline which generates two boxplots
      - **sqls**: directory containing SQL queries used to pull raw data from Vertica (most queries have been gitignored due to their sensitive nature)
      - **extras**
        - **utils**
        - **datasets**
          - **vertica_dataset.py**: extends the `AbstractDataSet` class to retrieve a SQL DataSet from a Vertica database
            
## Local Development

- Create a `local` directory within the `conf` directory
- Create a `credentials.yml` file to store all credentials you will need for local development. **NEVER** commit this file to GitHub.
- Entries for this example project are of the form:

```
prod_vertica_cred:
    server: aws_prod
    user: ${USERNAME}
    password: ${PASSWORDD}
```


## Data Processing

Typically it is useful to create a pipeline that performs all the data processing
steps needed to create the modeling dataset. 

- The Data Catalog provides a registry of all the data sources used in the project. This can be found in the `kedro-poc/conf/base/catalog.yml` file
- In the Data Catalog, we specify the name of each dataset, the type of dataset, and any other parameters required to create that type of dataset (i.e. a filepath for a CSV, for example).
- Next, data processing nodes are created in the `kedro-poc/src/kedro_poc/pipelines/data_processing/nodes.py` file.
- Nodes are simply python functions that perform some type of data processing or feature engineering steps on a given dataset or datasets, then return the processed data.
- Next the Data Processing Pipeline is configured in the `kedro-poc/src/kedro_poc/pipelines/data_processing/pipeline.py` file. The pipeline specifies the inputs and outputs of each node.
- We can register the outputs of data processing nodes in the Data Catalog as well, and this will ensure that they are saved as the specified file type after they are created by running that node.
- Kedro supports a large number of different data set types. More info on kedro dataset types can be found [here.](https://docs.kedro.org/en/stable/data/data_catalog.html)
- In this example, we extend kedro's `AbstractDataSet` type to create a `VerticaSQLQueryDataSet` class that queries data from Vertica using mmpac. This class definition can be found in `kedro-poc/src/kedro_poc/extras/datasets/vertica_dataset.py` 

Individual nodes in the data processing pipeline can be run using the following command: `kedro run --nodes=name_of_node`. 

## Modeling

Once we have completed our Data Processing pipeline, we can create our Modeling Pipeline, which 
will use the model data we created in order to train a data science model, or multiple models.

- With `kedro`, you can create modular pipelines that can be reused multiple times, with different inputs/outputs or parameters, within the same project, or shared across projects. 
- Modeling Nodes are created in the `kedro-poc/src/kedro_poc/pipelines/data_science/nodes.py` file. 
- Again, nodes are simply wrappers for python function that perform a step in the data science process. In this pipeline, we have a node to split the data into train, validation, and test sets, a node to train an XGBoost model, and a node to evaluate the model's performance on the test set.
- We build the Modeling Pipeline in `kedro-poc/src/kedro_poc/pipelines/data_science/pipeline.py` file.
- You can define parameters for machine learning models, like the train/test split and model-specific parameters, in the `conf/base/parameters/` directory.
- It is easy to run the entire pipeline from data processing to model training, or “slice” the pipeline, like just running the data science pipeline to perform hyperparameter tuning, for example.
- In the `data_science/yml` parameter file, you can see that two sets parameter values are specified to test different versions the XGBoost model. These are then each trained and evaluated [here](https://github.com/mbrody22/kedro-poc/blob/main/kedro-poc/src/kedro_poc/pipelines/data_science/pipeline.py#L35-L44).
- In this way, we only need to define the modeling pipeline once, but it can be instantiated multiple times with different inputs/outputs/parameters.

## Running Pipelines

To run any `kedro` commands, you must be in the `kedro_poc` directory. To run the entire pipeline, including data processing and
modeling, use the command `kedro run`. To run only one pipeline, like just the data science pipeline, use the command `kedro run --pipeline=data_science`.
To run a single node in a pipeline, use the command `kedro run --nodes=name_of_node`.


## Pipeline Visualization

Make sure you have `kedro-viz` installed. Running the command `kedro viz` will automatically open
a browser tab to serve the visualization of the entire pipeline. More info on kedro visualizations can be found [here](https://docs.kedro.org/en/stable/visualisation/kedro-viz_visualisation.html).
The visualization shows and Directed Acyclic Graph (DAG) of all the datasets that get pulled in, how they get split, how each data set gets used, and then the output of the model(s). Visualizations are also easy to share, as they can be saved as a JSON file or PNG. Below you can see the visualization of the pipeline created in this repo.

<img width="1049" alt="Screenshot 2023-06-06 at 8 59 24 AM" src="https://github.com/mbrody22/kedro-poc/assets/67015594/823c10c2-21f1-4f9c-b88c-2bb8839cc9be">

Clicking on an individual node will open a side panel that reveals more information about that node, including its file path, parameters, inputs, and outputs.

![Screenshot 2023-06-06 at 9 00 49 AM](https://github.com/mbrody22/kedro-poc/assets/67015594/5ef2322d-183b-4074-b76f-44383db4bc68)

## Plotting

Kedro viz can also be used to create other visualizations like plots. Kedro supports two Plotly data set types: plotly.PlotlyDataSet, which only support Plotly Express, and plotly.JSONDataSet, which supports Plotly Express and Plotly Graph Objects.

You can create a reporting pipeline by running the command: `kedro pipeline create reporting`. This will create a new `reporting` directory in your `pipelines` directory where you can define the nodes and pipeline for the reporting pipeline. Alternatively, you can just add nodes that output plots to you exising data prcoessing and/or data science pipeline(s).

Then, add an entry using these types of Plotly datasets to the Data Catalog for each type of graph you want to create. In this example repo, you can see that we create one Plot Express graph and one Plotly Graph Object. Next, define the nodes and pipeline for your reporting pipeline in the appropriate files in the new `reporting` directory.

Once you have defined your plots, you can run your reporting pipeline using the command: `kedro run --pipeline=reporting`. You can then view your plots by running `kedro viz`. The nodes and outputs you defined in your reporting pipeline will now be displayed in your pipeline DAG. You can click on the name of a plot in the diagram or in the menu on the left side of the screen to view your plot.

![Screenshot 2023-06-07 at 2 19 45 PM](https://github.com/mbrody22/kedro-poc/assets/67015594/43a4901e-dbdf-4339-8e80-ffe3ce60a1be)

Then click `Expand Plotly Visualization` to get a better look at your graph.

![Screenshot 2023-06-07 at 2 19 55 PM](https://github.com/mbrody22/kedro-poc/assets/67015594/fefe3ac3-f8e2-4580-b1fb-a38c2672af00)

Kedro also supports Matplotlib plots using the [MatplotlibWriter](https://docs.kedro.org/en/stable/kedro_datasets.matplotlib.MatplotlibWriter.html) dataset type, so you can create other types of figures like Confusions Matrices and many more. More info and examples can be found [here](https://docs.kedro.org/en/stable/visualisation/visualise_charts_with_plotly.html)

## Experiment Tracking

Experiment tracking is the process of saving all the metadata related to an experiment each time you run it. It enables you to compare different runs of a machine-learning model as part of the experimentation process. This metadata can be stored locally or remotely, such as in AWS S3.

In particular, each pipeline run during experiment tracking is considered a session. A session store records all related metadata such as a timestamp, performance metrics, parameter values, and even data sets used in the experiment run.
We do not implement experiment tracking in this repo, but more info on experiment tracking with Kedro and a short demo to set up tracking locally and in S3 can be found [here](https://docs.kedro.org/en/stable/experiment_tracking/index.html). 

Locally, a `SQLiteStore` is created in the `data` directory of the project. Then, nodes in the data science and data processing pipelines must be updated to return the particular outputs that we wish to track for our experiments. These outputs also need to be added to the catalog.yml as tracking.MetricsDataSets or tracking.JSONDataSets.

After running the pipeline, you can open the Kedro viz web app and click on an Experiment Tracking icon on the left to view the experiments that have been run and directly compare metrics and/or plots that were generated across different runs.
Kedro viz experiment tracking includes a time series feature so you can view how an individual metric has changed over time.

## Pros of Kedro

- You can create a new project from a config file, so we could create a base project config that could be used to quickly spin up new data science projects. 
- This would also ensure a consistent repo structure for data science projects.
- It is easy to include unit tests, as kedro projects are preconfigured to use `pytest`.
- Overall, it is easy to learn and pretty quick to implement projects in this format.
- The visualization feature is a really cool, interactive way to understand the entire data processing and modeling pipeline you have constructed and share it with others.
- Kedro Viz 6.2 includes support for collaborative experiment tracking using a cloud storage system like AWS S3, so multiple users can store their experiment data in a centralized remote location.
- You can combine MLflow with Kedro using `kedro-mlflow`, since Kedro alone does not support a Model Registry or model serving capabilities. 

## Cons of Kedro

- Supports Python but not R.
- Not useful for the data exploration phase of a project, just data processing to construct the model data and then model training, tuning and evaluation.
- Loading data from Vertica using SQL queries is not straightforward.
- Kedro does not support a Model Registry on its own.
