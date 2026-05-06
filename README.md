<div align="center">
  <h1>📊 JB Data Explorer</h1>
  <p><b>An Agent-Based Automated Enterprise Analytics & Reporting Platform</b></p>
  
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
  [![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75.svg?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
  [![OpenAI](https://img.shields.io/badge/AI-Powered-412991.svg?style=flat-square&logo=openai&logoColor=white)](https://openai.com/)
</div>

<br/>

## 📖 Overview

**JB Data Explorer** is a next-generation, AI-driven data analytics platform designed to bridge the gap between raw datasets and executive-level decision-making. 

Instead of writing manual Pandas code or configuring BI dashboards, the platform utilizes a **Multi-Agent Architecture** to autonomously ingest, clean, explore, model, and report on any tabular dataset (CSV or Excel). By combining rigorous statistical heuristics with Large Language Models (LLMs), JB Data Explorer acts as an automated Principal Data Scientist.

---

## ✨ Core Features

### 🧠 1. Multi-Agent AI Architecture
The platform delegates specialized tasks to a network of intelligent agents:
- **Domain Agent:** Analyzes the schema to automatically deduce the industry, business context, target variable, and problem type (Classification, Regression, etc.).
- **Cleaning Agent:** Executes a strict multi-step cleaning pipeline including dtype inference, duplicate removal, advanced imputation (mean/median/mode via skewness), and IQR-based winsorization.
- **Transformation Agent:** Prepares data for machine learning using Standard Scaling and Label Encoding.
- **ML Agent:** Runs an automated algorithm sweep across Random Forests, Logistic Regression, SVMs, Decision Trees, and XGBoost, constructing a performance leaderboard.
- **NLP Agent:** Translates plain-English user queries into Plotly Express visualizations accompanied by rich data narratives.
- **Report Narrator:** Drafts strictly formatted, highly descriptive executive summaries and conclusions.

### 📈 2. Automated Exploratory Data Analysis (EDA)
Automatically generates a suite of interactive visualizations:
- Feature distributions and class balances.
- Missing value heatmaps.
- Correlation matrices with AI-generated interpretations.
- Scatter and box plots for numeric/categorical relationships.

### 🗣️ 3. Natural Language Querying (Semantic API)
Ask questions like *"Show me the revenue breakdown by region"* and the NLP Agent will:
1. Filter and aggregate the data via generated Pandas logic.
2. Select the optimal Plotly chart type (Bar, Line, Scatter, Pie, Treemap, etc.).
3. Apply a vibrant, modern color sequence.
4. Generate a professional caption and a descriptive narrative explaining the insights.

### 📄 4. Consulting-Grade PDF Report Architect
Export your entire analytical session into a stunning, template-driven A4 PDF.
- **Two-Pass Generation:** Automatically calculates page numbers to build a perfectly accurate **Table of Contents**.
- **Embedded Graphics:** High-resolution Plotly charts seamlessly embedded via Kaleido.
- **Dynamic Text Generation:** Extensively detailed AI narratives (strict 18-sentence executive summaries, deep ML interpretations).
- **Audit Logging:** An appendix containing a complete, emoji-free pipeline execution trace for total transparency.

---

## 🛠️ Technology Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Data Processing:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Data Visualization:** Plotly Express, Kaleido (for static image export)
- **PDF Generation:** FPDF2
- **LLM Integration:** OpenAI SDK (configured for GitHub Models / Azure Inference endpoints)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- A GitHub Personal Access Token (PAT) or OpenAI API Key.

### 1. Clone the Repository
```bash
git clone https://github.com/JB14forever/data-analytics-platform.git
cd data-analytics-platform
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
The application uses the `utils.llm_client` to communicate with the LLM. You must provide an API key. Create a `.env` file in the root directory:

```env
# Primary: GitHub Models Endpoint (Free Tier)
GITHUB_TOKEN=ghp_your_github_personal_access_token

# Fallback: Standard OpenAI API Key
OPENAI_API_KEY=sk-your_openai_api_key
```

### 4. Launch the Platform
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
data_analytics_platform/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python package dependencies
├── .env                        # Environment variables (not tracked in git)
├── agents/                     # Intelligent Agent Modules
│   ├── ingestion_agent.py      # Handles file uploading and basic parsing
│   ├── domain_agent.py         # AI domain context and schema deduction
│   ├── cleaning_agent.py       # Heuristic and statistical data cleaning
│   ├── transformation_agent.py # ML preprocessing (Scaling, Encoding)
│   ├── ml_agent.py             # Automated model training and evaluation
│   ├── nlp_agent.py            # Natural Language to Graph Semantic Engine
│   ├── graph_describer.py      # Generates descriptive text for EDA charts
│   └── report_narrator.py      # Drafts executive summaries and PDF content
└── utils/                      # Shared Utilities
    ├── helpers.py              # UI rendering and layout constants
    ├── llm_client.py           # Centralized OpenAI/GitHub Models client
    └── pdf_generator.py        # FPDF2 engine with Two-Pass TOC rendering
```

---

## 📝 Recent Updates & Enhancements
- **Intelligent Table of Contents:** Implemented a two-pass rendering algorithm in `pdf_generator.py` to calculate exact page numbers for dynamic sections.
- **Strict Narrative Constraints:** Upgraded `ReportNarrator` and `DomainAgent` prompts to enforce exact sentence counts (e.g., 18 sentences, 3 paragraphs) for incredibly deep, professional reports.
- **UI & Margin Fixes:** Removed restrictive column layouts from the NLP interface and fixed FPDF X-coordinate margin bleeding to ensure text never overlaps with visual elements.
- **Codebase Optimization:** Cleaned up unused library imports and stabilized the `requirements.txt` environment.

---

<div align="center">
  <i>Built with ❤️ for enterprise-grade analytics automation.</i>
</div>
