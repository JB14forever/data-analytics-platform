# D:\data_analytics_platform\agents\domain_agent.py

import os
import json
import streamlit as st
from openai import OpenAI

class DomainAgent:
    """
    Leverages OpenAI to understand industry context and recommend configurations.
    """
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except (KeyError, FileNotFoundError):
                pass
                
        self.available = bool(api_key)
        self.client = OpenAI(api_key=api_key) if self.available else None
        
    def analyze_context(self, schema: dict, sample_data: list) -> dict:
        """
        Takes schema and row sample, asks LLM to identify the dataset.
        Returns JSON structure: {
            "industry": string,
            "target_variable": string,
            "evaluation_metric": string,
            "business_summary": string,
            "suggested_queries": list[string]
        }
        """
        if not self.available:
            return self._fallback_context(schema)
            
        system_prompt = \"\"\"
        You are a Principal Data Scientist advising an enterprise analytics platform.
        You will be provided with a JSON schema of a dataset and a small sample of rows.
        
        Analyze the data and output ONLY a raw JSON string matching exactly this format (no markdown tags):
        {
            "industry": "Extracted Industry Context (e.g. Healthcare, Finance, Sports, Sales...)",
            "target_variable": "The exact name of the column mathematically best suited to be the ML prediction target. Look for status, price, salary, churn, diagnosis, outcome. ONLY string from the schema keys.",
            "evaluation_metric": "The best metric to evaluate this dataset's target (e.g. F1-Score, RMSE, ROC-AUC, MAE) and why (1 short sentence).",
            "business_summary": "A 2-sentence summary of what this dataset appears to be modeling.",
            "suggested_queries": [
                "Suggest 1 basic analytical natural language query",
                "Suggest 1 advanced insight-driven natural language query"
            ]
        }
        \"\"\"
        
        user_prompt = f"SCHEMA: {json.dumps(schema)}\nSAMPLE ROWS: {json.dumps(sample_data, default=str)}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()
            
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
                
            return json.loads(raw)
            
        except Exception:
            return self._fallback_context(schema)
            
    def _fallback_context(self, schema: dict) -> dict:
        """Heuristic fallback."""
        possible_targets = [c for c in schema.keys() if any(x in str(c).lower() for x in ['target', 'is_', 'status', 'churn', 'price', 'salary', 'outcome'])]
        target = possible_targets[0] if possible_targets else list(schema.keys())[-1]
        
        return {
            "industry": "General Business",
            "target_variable": target,
            "evaluation_metric": "Accuracy / RMSE depending on classification or regression.",
            "business_summary": "Auto-identified schema without semantic domain awareness (API Missing).",
            "suggested_queries": [
                f"Show distribution of {target}."
            ]
        }
