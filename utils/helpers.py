# D:\data_analytics_platform\utils\helpers.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def render_health_badge(score: float) -> str:
    if score >= 80:
        color = "#2e7d32"
        text = "Excellent"
    elif score >= 50:
        color = "#ed6c02"
        text = "Needs Review"
    else:
        color = "#d32f2f"
        text = "Critical"
        
    return f'''
    <div style="display: inline-block; padding: 0.3em 0.8em; font-size: 85%; font-weight: 600; 
        border-radius: 4px; background-color: {color}; color: white; letter-spacing: 0.5px;">
        {score}/100 - {text}
    </div>
    '''

def get_minimalist_layout():
    """Returns a minimalist, aesthetically pleasing layout configuration for Plotly."""
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color="#4A4A4A"),
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#E0E0E0"),
        yaxis=dict(showgrid=True, gridcolor="#F5F5F5", zeroline=False, linecolor="#E0E0E0"),
        title_font=dict(size=18, color="#2C3E50", family="Inter, sans-serif")
    )

def df_to_plotly_heatmap(df: pd.DataFrame) -> go.Figure:
    import numpy as np
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty or numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for correlation heatmap.")
        return fig
        
    corr = numeric_df.corr().round(2)
    
    # Use a subtle minimalist sequential palette
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Teal",
        title="Feature Correlation Matrix"
    )
    fig.update_layout(**get_minimalist_layout())
    return fig

def df_to_plotly_histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(
        df, 
        x=col, 
        marginal="box",
        title=f"Distribution Analysis: {col}",
        color_discrete_sequence=['#3b82f6'], # Soft pleasant blue
        opacity=0.8
    )
    # Fix layout
    fig.update_layout(**get_minimalist_layout())
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return fig

def plotly_to_image_bytes(fig: go.Figure) -> bytes:
    """Exports a plotly figure to PNG bytes using kaleido for PDF embedding."""
    try:
        # scale=2 for retina quality
        return fig.to_image(format="png", engine="kaleido", scale=2)
    except Exception as e:
        print(f"Kaleido export error: {e}")
        return None

def apply_nlp_filter(df: pd.DataFrame, filter_code: str) -> pd.DataFrame:
    if not filter_code:
        return df
        
    try:
        filtered_df = eval(filter_code, {"pd": pd}, {"df": df})
        if isinstance(filtered_df, pd.DataFrame):
            return filtered_df
    except Exception:
        pass
        
    return df
