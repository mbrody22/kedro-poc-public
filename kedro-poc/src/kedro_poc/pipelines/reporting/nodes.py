import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

def age_plot_px(model_data: pd.DataFrame):
    '''Plot the age distributions of those who did and didn't buy the target product

        Args: pd.DataFrame of the pre-processed model_data with target column

        Returns: Plotly Express Object
        '''
    df = model_data[['age', 'target']]
    df['target'] = df['target'].astype('category')
    return df

def age_plot_go(model_data: pd.DataFrame):
    '''Plot the age distributions of those who did and didn't buy the target product

    Args: pd.DataFrame of the pre-processed model_data with target column

    Returns: Plotly Graph Object
    '''
    target1 = model_data.loc[model_data['target'] == 1].copy()
    target0 = model_data.loc[model_data['target'] == 0].copy()

    fig = go.Figure(layout=go.Layout(
        title=go.layout.Title(text="Client Age by DI Ownership Status")
    ))

    fig.add_trace(go.Box(
        y=target0['age'],
        name="Doesn't Own DI"
    ))

    fig.add_trace(go.Box(
        y=target1['age'],
        name='Owns DI'
    ))

    return fig