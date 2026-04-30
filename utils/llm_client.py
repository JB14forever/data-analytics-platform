# D:\data_analytics_platform\utils\llm_client.py

"""
Centralized LLM Client Factory
===============================
Single source of truth for all LLM interactions across the platform.
Uses the GitHub Models free-tier inference endpoint with a GitHub PAT.
No agent should ever import OpenAI directly — always use this module.
"""

import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file at module level
load_dotenv()

# ── Constants ──────────────────────────────────────────────────────
LLM_MODEL = "gpt-4o-mini"
_BASE_URL = "https://models.inference.ai.azure.com"

# ── Private token resolver ─────────────────────────────────────────
def _resolve_token() -> str | None:
    """
    Resolves the GitHub token using a secure fallback chain:
      1. Environment variable (GITHUB_TOKEN)
      2. Streamlit secrets (for cloud deployments)
    Never returns the token directly to callers outside this module.
    """
    token = os.getenv("GITHUB_TOKEN")
    if token: return token
    
    token = os.getenv("OPENAI_API_KEY")
    if token: return token

    # Fallback: Streamlit Secrets (for Streamlit Cloud deployments)
    try:
        if "GITHUB_TOKEN" in st.secrets:
            return st.secrets["GITHUB_TOKEN"]
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError, AttributeError):
        pass

    return None


# ── Public API ─────────────────────────────────────────────────────
def is_llm_available() -> bool:
    """Check if the LLM backend is configured and reachable."""
    return _resolve_token() is not None


def get_llm_client() -> OpenAI | None:
    """
    Returns a configured OpenAI client pointed at the GitHub Models
    inference endpoint. Returns None if no token is available.
    
    Usage:
        from utils.llm_client import get_llm_client, LLM_MODEL
        client = get_llm_client()
        if client:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[...]
            )
    """
    token = _resolve_token()
    if not token:
        return None

    return OpenAI(
        base_url=_BASE_URL,
        api_key=token
    )
