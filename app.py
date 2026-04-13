# D:\data_analytics_platform\app.py

import streamlit as st
import pandas as pd
import json

from agents.ingestion_agent import IngestionAgent
from agents.cleaning_agent import CleaningAgent
from agents.transformation_agent import TransformationAgent
from agents.ml_agent import MLAgent
from agents.domain_agent import DomainAgent
from agents.nlp_agent import NLPAgent

from utils.pdf_generator import generate_pdf
from utils.helpers import (
    render_health_badge, 
    df_to_plotly_heatmap, 
    df_to_plotly_histogram, 
    apply_nlp_filter,
    plotly_to_image_bytes
)

# ==========================================
# 1. Page Config
# ==========================================
st.set_page_config(
    page_title="Automated Analytics v2",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. Session State Initialization
# ==========================================
STATE_KEYS = {
    'raw_df': None,
    'clean_df': None,
    'transformed_df': None,
    'schema': {},
    'domain_context': {},
    'cleaning_logs': {},
    'health_score': 0.0,
    'ml_results': {},
    'saved_nlp_queries': [], 
    'eda_images_bytes': [],
    'pipeline_stage': 'Awaiting Data',
    
    # Metadata
    'author_name': '',
    'project_title': 'Analytics Platform V2',
    'project_desc': '',
    'dataset_desc': ''
}

for k, v in STATE_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 3. Sidebar Configuration
# ==========================================
with st.sidebar:
    st.title("⚙️ Report Configuration")
    
    # User / Report Params
    st.session_state['project_title'] = st.text_input("Project Title", value=st.session_state['project_title'])
    st.session_state['author_name'] = st.text_input("Author Name", placeholder="e.g. Jane Doe")
    st.session_state['project_desc'] = st.text_area("Project Description")
    st.session_state['dataset_desc'] = st.text_input("Dataset Context target (Optional)")
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload Dataset (CSV / XLXS)", type=["csv", "xlsx", "xls"])
    run_pipeline_btn = st.button("🚀 Execute Smart Pipeline", use_container_width=True)
    
    st.divider()
    st.subheader("💓 Dataset Vital Signs")
    
    df_ref = st.session_state['raw_df']
    rows = df_ref.shape[0] if df_ref is not None else 0
    cols = df_ref.shape[1] if df_ref is not None else 0
    
    st.metric("Rows", rows)
    st.metric("Columns", cols)
    st.markdown(f"**Stage:** `{st.session_state['pipeline_stage']}`")
    
    if st.session_state['pipeline_stage'] == 'Completed':
        st.divider()
        st.subheader("📥 Export Final Report")
        with st.spinner("Generating PDF Profile..."):
            pdf_b = generate_pdf(
                author_name=st.session_state['author_name'],
                dataset_name=uploaded_file.name if uploaded_file else "Data",
                project_desc=st.session_state['project_desc'],
                domain_context=st.session_state['domain_context'],
                cleaning_logs=st.session_state['cleaning_logs'],
                ml_results=st.session_state['ml_results'],
                eda_images=st.session_state['eda_images_bytes'],
                saved_queries=st.session_state['saved_nlp_queries']
            )
        st.download_button(
            label="Download Complete PDF",
            data=pdf_b,
            file_name="Automated_Analytics_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ==========================================
# Execution Engine Pipeline
# ==========================================
if run_pipeline_btn and uploaded_file:
    try:
        # Phase 1: Ingestion
        with st.spinner("Agent 1/4: Ingesting & Heuristic Checks..."):
            ingestor = IngestionAgent()
            raw_data = ingestor.load_data(uploaded_file)
            st.session_state['raw_df'] = raw_data
            
            # Filter primary keys and Zero variance
            filtered_df, drops_1 = ingestor.filter_primary_features(raw_data)
            st.session_state['health_score'] = ingestor.compute_health_score(filtered_df)
            st.session_state['pipeline_stage'] = 'Ingestion Finished'
            
        # Phase 2: Domain Context Identification
        with st.spinner("Agent 2/4: OpenAI Context Identification..."):
            initial_schema = ingestor.infer_schema(filtered_df)
            sample_rows = filtered_df.head(5).to_dict(orient='records')
            
            domain = DomainAgent()
            context = domain.analyze_context(initial_schema, sample_rows)
            st.session_state['domain_context'] = context
            st.session_state['pipeline_stage'] = 'Context Resolved'
            
        # Phase 3: Rigorous 12-Step Cleaning
        with st.spinner("Agent 3/4: Deep Cleaning (Missing, Outliers, Bounds)..."):
            cleaner = CleaningAgent()
            clean_df, miss_drops, dups_removed = cleaner.clean(filtered_df)
            
            # Aggregate cleaning logs for PDF
            st.session_state['cleaning_logs'] = {**drops_1, **miss_drops}
            if dups_removed > 0:
                st.session_state['cleaning_logs']['System_Dups'] = f"Removed {dups_removed} duplicate rows."
                
            st.session_state['clean_df'] = clean_df
            
            # WE MUST Re-infer schema because cleaning agent standardizes column names!
            clean_schema = ingestor.infer_schema(clean_df)
            st.session_state['schema'] = clean_schema
            st.session_state['pipeline_stage'] = 'Cleaning Done'
            
        # Phase 4: Transformation for ML
        with st.spinner("Agent 4/4: Feature Enc & Scale..."):
            transformer = TransformationAgent()
            st.session_state['transformed_df'] = transformer.transform(st.session_state['clean_df'], clean_schema)
            # Re-verify target variable maps correctly just in case standardization broke the case
            target = context.get('target_variable', "")
            clean_cols = st.session_state['transformed_df'].columns
            if target and target.lower() in [x.lower() for x in clean_cols]:
                true_target = [x for x in clean_cols if x.lower() == target.lower()][0]
                st.session_state['domain_context']['target_variable'] = true_target
                
            st.session_state['pipeline_stage'] = 'Completed'
            st.rerun()

    except Exception as e:
        st.sidebar.error(f"Execution Error: {e}")

# ==========================================
# Main Dashboard UI
# ==========================================
st.title(st.session_state['project_title'])

if st.session_state['raw_df'] is None:
    st.info("👈 Upload your dataset and execute the pipeline to begin.")
else:
    t1, t2, t3, t4, t5 = st.tabs(["🚀 Phase 1: Context", "🔍 Phase 2: EDA", "🤖 Phase 3: ML Modeling", "💬 Phase 4: Query Insights", "📑 Report Architect"])
    
    # -------------------------------------
    # TAB 1: Profile & Context
    # -------------------------------------
    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Data Health & Overview")
            st.markdown(render_health_badge(st.session_state['health_score']), unsafe_allow_html=True)
            st.dataframe(st.session_state['clean_df'].head(50), use_container_width=True)
        with c2:
            ctx = st.session_state['domain_context']
            if ctx:
                st.subheader("🧠 OpenAI Domain Resolution")
                st.info(f"**Extrapolated Industry:** {ctx.get('industry', 'N/A')}")
                st.success(f"**Identified Core ML Target:** `{ctx.get('target_variable', 'None')}`")
                st.warning(f"**Best Metric Strategy:** {ctx.get('evaluation_metric', 'N/A')}")
                st.markdown(f"*{ctx.get('business_summary', '')}*")
                
            st.subheader("🧹 Cleaning Modifications")
            if st.session_state['cleaning_logs']:
                st.json(st.session_state['cleaning_logs'])
            else:
                st.write("Dataset passed baseline checks un-scathed.")

    # -------------------------------------
    # TAB 2: Exploratory Data Analysis
    # -------------------------------------
    with t2:
        st.subheader("Statistical Distributions")
        c3, c4 = st.columns([1, 2])
        with c3:
            s_col = st.selectbox("Analyze Element:", st.session_state['clean_df'].columns)
            st.markdown("All generated graphs using this tool can be appended to your final offline PDF report automatically.")
            
        with c4:
            if s_col:
                fig_hist = df_to_plotly_histogram(st.session_state['clean_df'], s_col)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                if st.button(f"📸 Add {s_col} Distribution to Report"):
                    img_bytes = plotly_to_image_bytes(fig_hist)
                    if img_bytes:
                        st.session_state['eda_images_bytes'].append(img_bytes)
                        st.success("Graph Captured!")
                        
        st.divider()
        st.subheader("Macro Correlations")
        fig_heat = df_to_plotly_heatmap(st.session_state['clean_df'])
        st.plotly_chart(fig_heat, use_container_width=True)
        if st.button("📸 Add Heatmap to Report"):
            h_bytes = plotly_to_image_bytes(fig_heat)
            if h_bytes:
                st.session_state['eda_images_bytes'].append(h_bytes)
                st.success("Heatmap Captured!")

    # -------------------------------------
    # TAB 3: ML Algorithm Suite
    # -------------------------------------
    with t3:
        st.subheader("Model Prototyping Leaderboard")
        opt_target = st.session_state['domain_context'].get('target_variable', "")
        all_cols = list(st.session_state['transformed_df'].columns)
        
        # Ensure default target exists
        idx = all_cols.index(opt_target) if opt_target in all_cols else 0
        target_col = st.selectbox("Confirm Target Variable:", all_cols, index=idx)
        
        if st.button("🚂 Initiate Algorithm Sweeps"):
            with st.spinner("Agent Training Models (Multiple)..."):
                ml = MLAgent()
                try:
                    res = ml.train(st.session_state['transformed_df'], target_col)
                    st.session_state['ml_results'] = res
                except Exception as e:
                    st.error(f"Algorithm Sweep Failed: {e}")
                    
        mlr = st.session_state['ml_results']
        if mlr:
            st.success(f"Sweep Completed: **{mlr.get('task_type', '').capitalize()}** Problem.")
            st.markdown(f"**Dominant Engine:** {mlr.get('best_model_name')}")
            
            import plotly.express as px
            # Render Leaderboard
            df_board = pd.DataFrame(mlr.get('leaderboard', []))
            st.dataframe(df_board, use_container_width=True)
            
            st.markdown("### Structural Importance Extraction")
            fi = mlr.get('feature_importance', {})
            if fi:
                df_fi = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance']).sort_values(by='Importance')
                fig_fi = px.bar(df_fi, x='Importance', y='Feature', orientation='h', title="Feature Dependency Map")
                # Add layout properties
                from utils.helpers import get_minimalist_layout
                fig_fi.update_layout(**get_minimalist_layout())
                
                st.plotly_chart(fig_fi, use_container_width=True)
                
                if st.button("📸 Attach ML Dependency Plot"):
                    btn_bytes = plotly_to_image_bytes(fig_fi)
                    if btn_bytes:
                        st.session_state['eda_images_bytes'].append(btn_bytes)
                        st.success("Algorithm Plot Captured!")
                        
    # -------------------------------------
    # TAB 4: NLP Semantic Query API
    # -------------------------------------
    with t4:
        st.subheader("Semantic English-to-Graph Generation")
        st.markdown("Talk directly to your dataset. The system filters it safely behind the scenes.")
        
        sugg = st.session_state['domain_context'].get('suggested_queries', [])
        if sugg:
            st.caption("AI Ideas for this structure:")
            for s in sugg:
                st.markdown(f"- _{s}_")
                
        user_q = st.text_input("What do you want to see? (e.g., 'Chart revenue by user segments')")
        
        if st.button("Execute Natural Language") and user_q:
            nlp = NLPAgent()
            with st.spinner("Processing semantics..."):
                response = nlp.query(user_q, st.session_state['clean_df'], st.session_state['clean_df'].columns.tolist())
                
                if 'error' in response:
                    st.error(response['error'])
                else:
                    fc = response.get('filter_code')
                    ct = response.get('chart_type')
                    
                    df_v = apply_nlp_filter(st.session_state['clean_df'], fc)
                    st.code(fc if fc else "No sub-filtering applied.", language='python')
                    
                    current_fig = None
                    if ct:
                        from plotly import express as pxe
                        from utils.helpers import get_minimalist_layout
                        try:
                            # Heuristic assignments for axis
                            c_x = df_v.columns[0]
                            c_y = df_v.columns[1] if len(df_v.columns) > 1 else c_x
                            
                            c_t = ct.lower()
                            if c_t == 'bar':
                                current_fig = pxe.bar(df_v, x=c_x, y=c_y)
                            elif c_t == 'histogram':
                                current_fig = pxe.histogram(df_v, x=c_x)
                            elif c_t == 'line':
                                current_fig = pxe.line(df_v, x=c_x, y=c_y)
                            elif c_t == 'scatter':
                                current_fig = pxe.scatter(df_v, x=c_x, y=c_y)
                            elif c_t == 'box':
                                current_fig = pxe.box(df_v, x=c_x)
                                
                            if current_fig:
                                current_fig.update_layout(**get_minimalist_layout())
                                st.plotly_chart(current_fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not map chart cleanly: {e}")
                            
                    st.dataframe(df_v.head(20))
                    
                    # Store temporally
                    st.session_state['_last_nlp'] = {
                        'question': user_q,
                        'filter_logic': fc,
                        'current_fig': current_fig
                    }
                    
        # Check if there's a recent query to pin
        if '_last_nlp' in st.session_state:
            recent = st.session_state['_last_nlp']
            st.divider()
            if st.button("📌 Include This Specific Insight String in Final PDF"):
                fig_bytes = None
                if recent.get('current_fig'):
                    fig_bytes = plotly_to_image_bytes(recent['current_fig'])
                
                st.session_state['saved_nlp_queries'].append({
                    'question': recent['question'],
                    'filter_logic': recent['filter_logic'],
                    'image_bytes': fig_bytes
                })
                # clear active slot so we don't accidentally save twice
                del st.session_state['_last_nlp'] 
                st.success("Appended to Report Roster!")

    # -------------------------------------
    # TAB 5: Report Preview Architect
    # -------------------------------------
    with t5:
        st.subheader("A4 Export Verification")
        st.markdown("Ensure your arrays are populated before rendering the actual PDF.")
        st.write(f"**Queued EDA Graphics:** {len(st.session_state['eda_images_bytes'])} snapshots.")
        st.write(f"**Pinned Sentences/Queries:** {len(st.session_state['saved_nlp_queries'])} slots.")
        
        if st.session_state['ml_results']:
            st.write("✅ Machine Learning Models loaded.")
        else:
            st.write("❌ No ML Algorithm executed yet.")
