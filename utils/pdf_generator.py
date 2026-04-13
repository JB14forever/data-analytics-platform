# D:\data_analytics_platform\utils\pdf_generator.py

import datetime
from fpdf import FPDF
import io

def generate_pdf(
    author_name: str, 
    dataset_name: str, 
    project_desc: str, 
    domain_context: dict, 
    cleaning_logs: dict, 
    ml_results: dict, 
    eda_images: list, 
    saved_queries: list
) -> bytes:
    """
    Generates a highly structured, standard A4 PDF analytics report.
    """
    class PDF(FPDF):
        def header(self):
            # No header on first page
            if self.page_no() > 1:
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 10, f"{dataset_name} Analytics Report", border=False, align="R")
                self.ln(10)
            
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # ---------------------------------------------------------
    # TITLE PAGE
    # ---------------------------------------------------------
    pdf.set_y(60)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "Data Analytics Report", ln=True, align="C")
    
    pdf.set_font("Helvetica", "", 16)
    pdf.cell(0, 10, project_desc if project_desc else "Automated Platform Insights", ln=True, align="C")
    pdf.ln(20)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(52, 73, 94)
    # Metadata Block
    left_margin = 50
    pdf.set_x(left_margin)
    pdf.cell(40, 8, "Author:", border=0)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, author_name if author_name else "Unknown Analyst", ln=True)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(left_margin)
    pdf.cell(40, 8, "Dataset Name:", border=0)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, dataset_name, ln=True)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(left_margin)
    pdf.cell(40, 8, "Generated On:", border=0)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, datetime.datetime.now().strftime("%B %d, %Y - %H:%M"), ln=True)
    
    # Context Block
    if domain_context:
        pdf.ln(15)
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Domain Context", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        
        ind = str(domain_context.get('industry', 'General Business'))
        bs = str(domain_context.get('business_summary', ''))
        
        pdf.multi_cell(0, 6, f"Industry: {ind}", align="C")
        pdf.multi_cell(0, 6, bs, align="C")

    # ---------------------------------------------------------
    # SECTION 1: DATA CLEANING LOGS
    # ---------------------------------------------------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, "1. Data Cleaning Narrative", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    pdf.multi_cell(0, 6, "The dataset was subjected to a rigorous 12-step cleaning architecture, assessing missingness, enforcing distribution bounds via Winsorization, and filtering uninformative primary keys.")
    pdf.ln(5)
    
    if cleaning_logs:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Variables Dropped/Filtered:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for col, reason in cleaning_logs.items():
            pdf.cell(5, 6, "-")
            pdf.multi_cell(0, 6, f"[{col}]: {reason}")
    else:
        pdf.multi_cell(0, 6, "No severe structural anomalies were detected requiring column deletion.")
        
    pdf.ln(10)
    
    # ---------------------------------------------------------
    # SECTION 2: EXPLORATORY DATA ANALYSIS
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Exploratory Visualizations", ln=True)
    
    if eda_images:
        for img_bytes in eda_images:
            if img_bytes:
                img_io = io.BytesIO(img_bytes)
                # Ensure it fits on the page constraints nicely
                pdf.image(img_io, w=180)
                pdf.ln(10)
    else:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "No visual distributions were selected for the report.", ln=True)
        
    pdf.add_page()
    # ---------------------------------------------------------
    # SECTION 3: MACHINE LEARNING LEADERBOARD
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. Predictive Core Results", ln=True)
    
    if ml_results.get('leaderboard'):
        pdf.set_font("Helvetica", "", 11)
        target = domain_context.get('target_variable', 'Unknown Target') if domain_context else 'Target'
        task = ml_results.get('task_type', 'N/A').title()
        pdf.multi_cell(0, 6, f"An automated {task} sweep was conducted predicting '{target}'. The leaderboard reflects hold-out testing performance.")
        pdf.ln(5)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Model Leaderboard", ln=True)
        pdf.set_font("Helvetica", "B", 10)
        
        # Table Header
        col_w = 90
        pdf.cell(col_w, 8, "Algorithm", border=1)
        pdf.cell(col_w, 8, ml_results.get('metric_name', 'Score'), border=1)
        pdf.ln()
        
        pdf.set_font("Helvetica", "", 10)
        for entry in ml_results.get('leaderboard', []):
            pdf.cell(col_w, 8, str(entry.get('Model')), border=1)
            pdf.cell(col_w, 8, f"{entry.get(ml_results.get('metric_name'), 0):.4f}", border=1)
            pdf.ln()

        pdf.ln(10)
        
        if ml_results.get('feature_importance'):
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Top 10 Feature Importances", ln=True)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(90, 8, "Feature", border=1)
            pdf.cell(90, 8, "Relative Importance", border=1)
            pdf.ln()
            
            pdf.set_font("Helvetica", "", 10)
            for feat, score in ml_results.get('feature_importance', {}).items():
                pdf.cell(90, 8, str(feat)[:40], border=1)
                pdf.cell(90, 8, f"{score:.4f}", border=1)
                pdf.ln()
    else:
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "No models were computed.", ln=True)
        
    pdf.ln(10)

    # ---------------------------------------------------------
    # SECTION 4: NLP SAVED QUERIES
    # ---------------------------------------------------------
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. Derived Analyst Insights", ln=True)
    pdf.set_font("Helvetica", "", 11)
    
    if saved_queries:
        for idx, sq in enumerate(saved_queries):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 8, f"Query {idx+1}: {sq.get('question', '')}")
            pdf.set_font("Helvetica", "", 10)
            
            if sq.get('filter_logic'):
                pdf.multi_cell(0, 6, f"Data Subspace Definition: {sq['filter_logic']}")
                
            q_img = sq.get('image_bytes')
            if q_img:
                pdf.ln(3)
                pdf.image(io.BytesIO(q_img), w=160)
            pdf.ln(10)
    else:
        pdf.multi_cell(0, 8, "No natural language queries were attached to this report.")

    try:
        return bytes(pdf.output())
    except Exception:
        return pdf.output(dest='S').encode('latin1')
