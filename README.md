# 🚀 Automated Data Analytics Platform

An Agent-based automated Data Analytics Pipeline built with Streamlit and modern Python libraries. This platform follows the Single Responsibility Principle, dispatching specialized 'Agents' to handle distinct phases of the data science lifecycle.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange)

## ✨ Features and Architecture

The platform utilizes a modular architecture designed for high scalability and robustness:

### 1. 📥 Ingestion Agent
- **Smart Parsing**: Utilizes `csv.Sniffer` to bypass tedious delimiter configurations.
- **Profiling**: Automatically infers column data types, calculates column cardinality, and derives missingness.
- **Data Health Score**: Mathematically deduces an overall 'Vital Sign' badge assessing duplicate/null densities.

### 2. 🧹 Cleaning Agent
- **Adaptive Imputation**: Uses `scipy.stats.skew` to decide between Mean and Median imputation based on the distributional tail.
- **Outlier Caps**: Enforces extreme-value boundaries using IQR Winsorization.

### 3. ⚙️ Transformation Agent
- **Cardinality-Aware Encoding**: Protects against the Curse of Dimensionality by picking One-Hot Encoding for low variants and Label Encoding for high variants.
- **Scaling**: Standardizes numerical variance for ML-readiness.

### 4. 🤖 ML Agent 
- Automates Task Detection (Classification vs Regression).
- Automatically branches to Random Forest or XGBoost.
- Extracts optimized performance metrics and visualizes feature importances.

### 5. 💬 NLP Agent
- Connects to OpenAI (`gpt-4o-mini`).
- Safely processes Natural Language phrasing into strict Pandas execution layers.
- Supports conversational dashboard visualizations (Dynamic heatmaps, auto-binned histograms).

---

## 🏃‍♂️ Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/JB14forever/data-analytics-platform.git
   cd data-analytics-platform
   ```
2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API key in the environment:
   Create a `.env` file from the `.env.example` template:
   ```env
   OPENAI_API_KEY=your_key_here
   ```
4. Boot up the Streamlit UI:
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment

This project is configured to auto-deploy to **Streamlit Community Cloud** on pushes to the `main` branch. 
Ensure your `OPENAI_API_KEY` is securely stored under **Settings > Secrets** in the Streamlit Cloud Dashboard using the following format:

```toml
OPENAI_API_KEY = "sk-..."
```
