# D:\data_analytics_platform\agents\nlp_agent.py

import os
import json
import datetime
import pandas as pd
import streamlit as st
from openai import OpenAI


class NLPAgent:
    """
    Translates natural language questions into executable Pandas Code and Plot configurations.
    """

    def __init__(self):
        """
        Initializes the agent by securely loading the OpenAI API key.
        Checks environmental variables first, then Streamlit Secrets as a fallback.
        Gracefully degrades and flags itself unavailable if no key is found.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except (KeyError, FileNotFoundError):
                pass
                
        self.available = bool(api_key)
        self.client = OpenAI(api_key=api_key) if self.available else None

    def query(self, question: str, df: pd.DataFrame, columns: list) -> dict:
        """
        Queries the OpenAI API to translate natural language to data operations.
        
        Args:
            question (str): The user's typed question.
            df (pd.DataFrame): The dataframe context.
            columns (list): The list of columns available to the LLM.
            
        Returns:
            dict: Parsed JSON with 'filter_code' and 'chart_type' keys.
        """
        if not self.available:
            return {
                "filter_code": None,
                "chart_type": None,
                "error": "OpenAI API Key not found. Natural language queries disabled."
            }

        # Provide column names and partial dtypes to the LLM so it writes valid logic
        dtypes_info = {col: str(df[col].dtype) for col in columns if col in df.columns}
        
        system_prompt = f"""
        You are a seasoned Pandas Data Engineer.
        The user has a dataframe `df` with the following columns and dtypes:
        {json.dumps(dtypes_info)}
        
        Respond ONLY with a valid JSON object containing exactly two keys:
        - "filter_code": a string representing a valid single-line pandas operation 
          that filters the dataframe based on the user's question. Use `df` as the variable. 
          Return null if no filtering is required or possible.
        - "chart_type": a string representing the best way to visualize the answer. 
          Must be ONE of: 'bar', 'histogram', 'scatter', 'line', 'box', or null.
          
        Do not include markdown blocks, backticks, or other text outside the JSON.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.0
            )
            raw_content = response.choices[0].message.content.strip()
            
            # Clean up if model mistakenly returned markdown formatting
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
                
            parsed_json = json.loads(raw_content)
            
            return {
                "filter_code": parsed_json.get("filter_code"),
                "chart_type": parsed_json.get("chart_type")
            }
            
        except Exception as e:
            return {
                "filter_code": None,
                "chart_type": None,
                "error": f"Failed to process query: {str(e)}"
            }

    def log_query(self, question: str, response: dict):
        """
        Logs the user prompt and agent response to Streamlit session state.
        
        Args:
            question (str): The user's input.
            response (dict): The parsed response from the LLM.
        """
        if 'query_log' not in st.session_state:
            st.session_state['query_log'] = []
            
        st.session_state['query_log'].append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'question': question,
            'response': response
        })
