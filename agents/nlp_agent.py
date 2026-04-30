# D:\data_analytics_platform\agents\nlp_agent.py

import json
import datetime
import pandas as pd
import numpy as np
import streamlit as st
from utils.llm_client import get_llm_client, LLM_MODEL


class NLPAgent:
    """
    Intelligent Natural Language → Graph engine.
    Translates user questions into rich chart specifications with
    LLM-chosen chart types, proper labels, figure descriptions,
    and dataset-driven contextual narratives.
    """

    def __init__(self):
        self.client = get_llm_client()
        self.available = self.client is not None

    def query(self, question: str, df: pd.DataFrame, columns: list, domain_context: dict = None) -> dict:
        """
        Translates a natural language question into a full chart specification.
        
        Returns:
            dict with keys:
                - filter_code: pandas code to produce the data slice
                - chart_type: one of the supported plotly chart types
                - chart_config: dict with title, x, y, color, labels, legend_title
                - figure_description: professional caption for the chart
                - data_narrative: dataset-driven insight narrative
                - error: (only on failure)
        """
        if not self.available:
            return {
                "filter_code": None,
                "chart_type": None,
                "chart_config": {},
                "figure_description": None,
                "data_narrative": None,
                "error": "LLM API not configured. Natural language queries disabled."
            }

        # Build column context for the LLM
        dtypes_info = {}
        stats_info = {}
        for col in columns:
            if col not in df.columns:
                continue
            dtypes_info[col] = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                stats_info[col] = {
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    "mean": round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else None,
                    "unique_count": int(df[col].nunique())
                }
            else:
                top_vals = df[col].value_counts().head(8).to_dict()
                stats_info[col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in top_vals.items()}
                }

        domain_str = ""
        if domain_context:
            domain_str = f"""
            Domain Context:
            - Industry: {domain_context.get('industry', 'Unknown')}
            - Target Variable: {domain_context.get('target_variable', 'Unknown')}
            - Business Summary: {domain_context.get('business_summary', 'N/A')}
            """

        system_prompt = f"""You are an expert data visualization analyst. The user has a pandas DataFrame `df` with:

COLUMNS AND DTYPES:
{json.dumps(dtypes_info, indent=2)}

COLUMN STATISTICS:
{json.dumps(stats_info, indent=2, default=str)}

{domain_str}

TOTAL ROWS: {len(df)}

The user will ask a question in natural language. You must:
1. Choose the MOST SUITABLE chart type based on the data shape and question intent
2. Write valid pandas code to prepare the data for visualization
3. Provide professional chart attributes (title, axis labels, legend)
4. Write a figure description suitable for an analytical report
5. Write a data-driven narrative about what the data reveals — focus on patterns, distributions, comparisons, and insights from the actual dataset values. Do NOT explain why you chose a chart type.

SUPPORTED CHART TYPES (pick the most appropriate):
- "bar": categorical comparisons, rankings
- "histogram": single-variable distributions
- "line": trends over ordered/time data
- "scatter": relationship between two numeric variables
- "box": distribution spread and outliers
- "pie": proportional composition (use only when <=8 categories)
- "violin": distribution shape comparison across groups
- "area": cumulative trends
- "treemap": hierarchical proportions
- "funnel": sequential stage drop-off
- "sunburst": nested categorical hierarchies
- "heatmap": matrix correlations or pivot tables

Respond ONLY with a valid JSON object (no markdown, no backticks):
{{
    "filter_code": "valid single pandas expression using `df` that produces the data for charting, or null if using df directly. The result must be a DataFrame. For aggregations use .reset_index().",
    "chart_type": "one of the supported types above",
    "chart_config": {{
        "title": "A descriptive, professional chart title",
        "x": "column_name for x-axis",
        "y": "column_name for y-axis (null for histogram/pie if not needed)",
        "color": "column_name for color grouping or null",
        "labels": {{"original_col": "Human Readable Label"}},
        "legend_title": "Legend title or null"
    }},
    "figure_description": "A 2-3 sentence professional figure caption describing what the chart shows, suitable for an analytical report.",
    "data_narrative": "A 3-5 sentence narrative describing the key insights, patterns, distributions, or comparisons revealed by the data. Focus on actual data values, percentages, trends, and notable observations from the dataset."
}}"""

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            
            # Clean up markdown formatting if present
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            raw_content = raw_content.strip()
                
            parsed = json.loads(raw_content)
            
            return {
                "filter_code": parsed.get("filter_code"),
                "chart_type": parsed.get("chart_type"),
                "chart_config": parsed.get("chart_config", {}),
                "figure_description": parsed.get("figure_description"),
                "data_narrative": parsed.get("data_narrative"),
            }
            
        except Exception as e:
            return {
                "filter_code": None,
                "chart_type": None,
                "chart_config": {},
                "figure_description": None,
                "data_narrative": None,
                "error": f"Failed to process query: {str(e)}"
            }

    def log_query(self, question: str, response: dict):
        """
        Logs the user prompt and agent response to Streamlit session state.
        """
        if 'query_log' not in st.session_state:
            st.session_state['query_log'] = []
            
        st.session_state['query_log'].append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': question,
            'response': response
        })
