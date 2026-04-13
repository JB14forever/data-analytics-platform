# D:\data_analytics_platform\app.py

import streamlit as st
import pandas as pd
import plotly.express as px

from agents.ingestion_agent import IngestionAgent
from agents.cleaning_agent import CleaningAgent
from agents.transformation_agent import TransformationAgent
from agents.ml_agent import MLAgent
from agents.nlp_agent import NLPAgent

from utils.pdf_generator import generate_pdf
from utils.helpers import (
    render_health_badge, 
    df_to_plotly_heatmap, 
    df_to_plotly_histogram, 
    apply_nlp_filter
)

# ==========================================
# 1. Page Config
# ==========================================
# We initialize PROJECT_NAME early via session state to use it in page config if desired,
# though page config must be the first Streamlit command. We'll use a generic or session one.
default_project_name = "Analytics Platform"
if 'PROJECT_NAME' not in st.session_state:
    st.session_state['PROJECT_NAME'] = default_project_name

st.set_page_config(
    page_title=st.session_state['PROJECT_NAME'],
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Session State Initialization
# ==========================================
def init_session_state():
    state_keys = {
        'raw_df': None,
        'clean_df': None,
        'transformed_df': None,
        'schema': {},
        'health_score': 0.0,
        'quality_issues': [],
        'ml_results': {},
        'query_log': [],
        'pipeline_stage': 'Awaiting Data',
        'nlp_agent_instance': NLPAgent()
    }
    for key, default_val in state_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_val

init_session_state()

# ==========================================
# 3. Sidebar Construction
# ==========================================
with st.sidebar:
    st.title("⚙️ Configuration")
    project_name = st.text_input("Project Name", value=st.session_state['PROJECT_NAME'])
    # Update title dynamically on reload
    st.session_state['PROJECT_NAME'] = project_name
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    run_pipeline_btn = st.button("🚀 Run Full Pipeline", use_container_width=True)
    
    st.divider()
    
    st.subheader("💓 Dataset Vital Signs")
    
    # Calculate Display Metrics safely
    df_ref = st.session_state['raw_df']
    rows = df_ref.shape[0] if df_ref is not None else 0
    cols = df_ref.shape[1] if df_ref is not None else 0
    
    if df_ref is not None:
        null_pct = (df_ref.isnull().sum().sum() / (rows * cols)) * 100
    else:
        null_pct = 0.0
        
    st.metric("Rows", rows)
    st.metric("Columns", cols)
    st.metric("Global Null %", f"{null_pct:.2f}%")
    
    st.markdown("**Health Score:**")
    st.markdown(render_health_badge(st.session_state['health_score']), unsafe_allow_html=True)
    
    st.markdown(f"**Stage:** `{st.session_state['pipeline_stage']}`")
    
    st.divider()
    
    st.subheader("📥 Export")
    if st.session_state['raw_df'] is not None and st.session_state['pipeline_stage'] == 'Completed':
        # Prepare context for PDF
        summary_stats = {
            'rows': rows,
            'cols': cols,
            'null_percentage': null_pct,
            'numeric_stats': st.session_state['raw_df'].describe().to_dict() if not st.session_state['raw_df'].empty else {}
        }
        
        with st.spinner("Generating PDF..."):
            pdf_bytes = generate_pdf(
                project_name=st.session_state['PROJECT_NAME'],
                summary_stats=summary_stats,
                health_score=st.session_state['health_score'],
                schema=st.session_state['schema'],
                ml_results=st.session_state['ml_results'],
                query_log=st.session_state['query_log']
            )
            
        st.download_button(
            label="Download Full Analysis",
            data=pdf_bytes,
            file_name="analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )


# ==========================================
# Execution Logic (Sidebar triggered)
# ==========================================
if run_pipeline_btn:
    if uploaded_file is None:
        st.sidebar.error("Please upload a CSV file first.")
    else:
        try:
            # Phase 1: Ingestion
            with st.spinner("Agent 1/5: Ingesting & Profiling..."):
                ingestor = IngestionAgent()
                st.session_state['raw_df'] = ingestor.load_csv(uploaded_file)
                st.session_state['schema'] = ingestor.infer_schema(st.session_state['raw_df'])
                st.session_state['health_score'] = ingestor.compute_health_score(st.session_state['raw_df'])
                st.session_state['quality_issues'] = ingestor.flag_quality_issues(st.session_state['raw_df'])
                st.session_state['pipeline_stage'] = 'Profiling Done'
                
            # Phase 2: Cleaning
            with st.spinner("Agent 2/5: Cleaning & Imputing..."):
                cleaner = CleaningAgent()
                st.session_state['clean_df'] = cleaner.clean(st.session_state['raw_df'], st.session_state['schema'])
                st.session_state['pipeline_stage'] = 'Cleaning Done'
                
            # Phase 3: Transformation
            with st.spinner("Agent 3/5: Encoding & Scaling..."):
                transformer = TransformationAgent()
                st.session_state['transformed_df'] = transformer.transform(st.session_state['clean_df'], st.session_state['schema'])
                st.session_state['pipeline_stage'] = 'Transformation Done'
                
            # Phase 4: Machine Learning 
            # We defer ML until the user selects a target in the ML tab, OR we try to predict the last column.
            # To be robust, let's just mark it complete and run ML interactively in the tab.
            st.session_state['pipeline_stage'] = 'Completed'
            st.rerun()

        except Exception as e:
            st.sidebar.error(f"Pipeline Error: {e}")


# ==========================================
# 4. Main Area Layout
# ==========================================
st.title(st.session_state['PROJECT_NAME'])

if st.session_state['raw_df'] is None:
    st.info("👈 Upload a CSV and click 'Run Full Pipeline' to begin.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Analysis", "🤖 ML Results", "💬 NLP Query"])
    
    # ------------------
    # TAB 1: Overview
    # ------------------
    with tab1:
        st.subheader("Data Preview")
        st.dataframe(st.session_state['raw_df'].head(100), use_container_width=True)
        
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            st.subheader("Inferred Schema")
            # Convert schema dict to dataframe for nice rendering
            schema_df = pd.DataFrame.from_dict(st.session_state['schema'], orient='index')
            st.dataframe(schema_df, use_container_width=True)
            
        with col_s2:
            st.subheader("Quality Warnings")
            if not st.session_state['quality_issues']:
                st.success("No critical quality issues detected!")
            else:
                for issue in st.session_state['quality_issues']:
                    st.warning(issue)
                    
    # ------------------
    # TAB 2: Analysis
    # ------------------
    with tab2:
        st.subheader("Correlation Heatmap")
        st.plotly_chart(df_to_plotly_heatmap(st.session_state['clean_df']), use_container_width=True)
        
        st.divider()
        st.subheader("Feature Distributions")
        selected_col = st.selectbox("Select a column to view distribution:", st.session_state['clean_df'].columns)
        if selected_col:
            st.plotly_chart(df_to_plotly_histogram(st.session_state['clean_df'], selected_col), use_container_width=True)
            
    # ------------------
    # TAB 3: ML Results
    # ------------------
    with tab3:
        st.subheader("Automated Predictive Modeling")
        st.markdown("Select a target variable to trigger automated feature engineering and model training.")
        
        target_options = st.session_state['clean_df'].columns.tolist()
        target_col = st.selectbox("Target Column:", target_options, index=len(target_options)-1 if target_options else 0)
        
        if st.button("🚀 Train Model"):
            with st.spinner("Agent 4/5: Training ML Models..."):
                ml_agent = MLAgent()
                # Run on transformed df for best results
                try:
                    res = ml_agent.train(st.session_state['transformed_df'], target_col)
                    st.session_state['ml_results'] = res
                except Exception as e:
                    st.error(f"Modeling failed: {e}")

        res = st.session_state.get('ml_results', {})
        if res:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Task Type", res.get('task_type', '').capitalize())
            mc2.metric("Best Model", res.get('best_model_name', ''))
            mc3.metric(res.get('metric_name', 'Metric Value'), res.get('best_metric_value', ''))
            
            st.markdown("### Feature Importance")
            fi_p = res.get('feature_importance', {})
            if fi_p:
                fi_df = pd.DataFrame(list(fi_p.items()), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=True)
                fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', template='plotly_dark')
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")
                
    # ------------------
    # TAB 4: NLP Query
    # ------------------
    with tab4:
        st.subheader("Natural Language Querying")
        nlp_agent = st.session_state['nlp_agent_instance']
        
        if not nlp_agent.available:
            st.warning("⚠️ OpenAI API Key not configured. NLP features are disabled.")
            
        nlp_target_col = st.selectbox("Select context column (Optional):", st.session_state['clean_df'].columns, key="nlp_target")
        query_input = st.text_input("Ask a question about your data (e.g., 'Show me sales over 500 in a bar chart'):")
        
        if st.button("Submit Query") and query_input:
            with st.spinner("Agent 5/5: Analyzing query..."):
                cols_context = st.session_state['clean_df'].columns.tolist()
                response = nlp_agent.query(query_input, st.session_state['clean_df'], cols_context)
                nlp_agent.log_query(query_input, response)
                
                if 'error' in response:
                    st.error(response['error'])
                else:
                    st.success("Query processed safely without modifying the base state.")
                    
                    filter_code = response.get('filter_code')
                    chart_type = response.get('chart_type')
                    
                    # Apply filter
                    view_df = apply_nlp_filter(st.session_state['clean_df'], filter_code)
                    
                    if filter_code:
                        st.code(f"Applied Logic: {filter_code}", language='python')
                        
                    st.dataframe(view_df.head(50), use_container_width=True)
                    
                    # Apply Chart
                    if chart_type and nlp_target_col:
                        try:
                            valid_chart = chart_type.lower().strip()
                            if valid_chart == 'histogram':
                                fig = px.histogram(view_df, x=nlp_target_col, template='plotly_dark')
                            elif valid_chart == 'bar':
                                fig = px.bar(view_df, x=view_df.index, y=nlp_target_col, template='plotly_dark')
                            elif valid_chart == 'line':
                                fig = px.line(view_df, x=view_df.index, y=nlp_target_col, template='plotly_dark')
                            elif valid_chart == 'scatter':
                                # Requires 2 dims usually, we default x to index for simplicity
                                fig = px.scatter(view_df, x=view_df.index, y=nlp_target_col, template='plotly_dark')
                            elif valid_chart == 'box':
                                fig = px.box(view_df, y=nlp_target_col, template='plotly_dark')
                            else:
                                st.warning(f"Unrecognized Chart Type: {chart_type}")
                                fig = None
                                
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not render chart '{chart_type}': {e}")
                            
        st.divider()
        with st.expander("Show Query Log"):
            st.json(st.session_state['query_log'])
