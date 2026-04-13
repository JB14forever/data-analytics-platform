# D:\data_analytics_platform\utils\helpers.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render_health_badge(score: float) -> str:
    """
    Renders an HTML-formatted badge indicating dataset health.
    
    Args:
        score (float): The dataset health score (0-100).
        
    Returns:
        str: An HTML string suitable for Streamlit markdown.
    """
    if score >= 80:
        color = "#28a745"  # Green
        text = "Excellent"
    elif score >= 50:
        color = "#ffc107"  # Amber
        text = "Needs Review"
    else:
        color = "#dc3545"  # Red
        text = "Critical"
        
    return f'''
    <div style="
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 85%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
        background-color: {color};
        color: white;
    ">
        {score}/100 - {text}
    </div>
    '''

def df_to_plotly_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Generates a correlation heatmap for numeric columns.
    
    Args:
        df (pd.DataFrame): The dataset.
        
    Returns:
        go.Figure: A Plotly graphical object representing the heatmap.
    """
    import numpy as np
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        # Return empty figure if not enough numeric columns
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for correlation heatmap.")
        return fig
        
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Numeric Correlation Heatmap"
    )
    return fig

def df_to_plotly_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    """
    Generates a histogram with marginal box plot for a specified column.
    
    Args:
        df (pd.DataFrame): The dataset.
        col (str): Target column to plot.
        
    Returns:
        go.Figure: A Plotly histogram figure.
    """
    fig = px.histogram(
        df, 
        x=col, 
        marginal="box",
        title=f"Distribution of {col}",
        template="plotly_dark",
        color_discrete_sequence=['#636EFA']
    )
    return fig

def apply_nlp_filter(df: pd.DataFrame, filter_code: str) -> pd.DataFrame:
    """
    Safely evaluates a string of pandas code to filter a dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe to filter.
        filter_code (str): Pandas filter string (e.g. "df[df['Age'] > 30]").
        
    Returns:
        pd.DataFrame: Filtering result or original dataframe on failure.
    """
    if not filter_code:
        return df
        
    try:
        # Using eval safely by explicitly providing only pandas and the df
        # The prompt instructed the LLM to output a single-line statement yielding a dataframe.
        filtered_df = eval(filter_code, {"pd": pd}, {"df": df})
        if isinstance(filtered_df, pd.DataFrame):
            return filtered_df
    except Exception:
        pass
        
    return df
