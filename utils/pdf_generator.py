# D:\data_analytics_platform\utils\pdf_generator.py

import datetime
from fpdf import FPDF


def generate_pdf(project_name: str, summary_stats: dict, health_score: float, schema: dict, ml_results: dict, query_log: list) -> bytes:
    """
    Generates an executive summary PDF of the data analytics pipeline.
    
    Args:
        project_name (str): The name of the project.
        summary_stats (dict): General metrics like row/col count and null %.
        health_score (float): Calculated health score.
        schema (dict): Column schema definitions.
        ml_results (dict): Results from ML Agent.
        query_log (list): Log of NLP queries.
        
    Returns:
        bytes: The raw byte content of the generated PDF.
    """
    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 12)
            self.cell(0, 10, f"{project_name} - Analytics Report", border=False, align="C")
            self.ln(10)
            
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Title Page section
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Data Analytics Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 10, "Automated Agent-Based Analytics Platform", ln=True, align="C")
    pdf.ln(15)

    # Section 1 - Dataset Overview
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Dataset Overview", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Overall Health Score: {health_score}/100", ln=True)
    pdf.cell(0, 8, f"Total Rows: {summary_stats.get('rows', 0)}", ln=True)
    pdf.cell(0, 8, f"Total Columns: {summary_stats.get('cols', 0)}", ln=True)
    pdf.cell(0, 8, f"Global Null Percentage: {summary_stats.get('null_percentage', 0):.2f}%", ln=True)
    pdf.ln(5)
    
    # Schema Table
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(60, 8, "Column", border=1)
    pdf.cell(40, 8, "Type", border=1)
    pdf.cell(40, 8, "Null %", border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    
    # Write at most 30 columns to prevent massive PDFs for wide tables
    for col, meta in list(schema.items())[:30]:
        pdf.cell(60, 8, str(col)[:25], border=1)
        pdf.cell(40, 8, str(meta['dtype']), border=1)
        pdf.cell(40, 8, f"{meta['null_percentage']:.1f}%", border=1)
        pdf.ln()
    if len(schema) > 30:
        pdf.cell(140, 8, f"... and {len(schema)-30} more columns", border=1, ln=True)
    
    pdf.ln(10)
    
    # Section 2 - Summary Statistics
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Summary Statistics (Numeric)", ln=True)
    pdf.ln(5)
    
    if summary_stats.get('numeric_stats'):
        pdf.set_font("Helvetica", "B", 10)
        col_w = 38
        pdf.cell(col_w, 8, "Column", border=1)
        pdf.cell(col_w, 8, "Mean", border=1)
        pdf.cell(col_w, 8, "Median", border=1)
        pdf.cell(col_w, 8, "Std", border=1)
        pdf.cell(col_w, 8, "Min/Max", border=1)
        pdf.ln()
        
        pdf.set_font("Helvetica", "", 10)
        for col, stats in list(summary_stats['numeric_stats'].items())[:20]: # Limit to 20
            pdf.cell(col_w, 8, str(col)[:18], border=1)
            pdf.cell(col_w, 8, f"{stats.get('mean', 0):.2f}", border=1)
            pdf.cell(col_w, 8, f"{stats.get('median', 0):.2f}", border=1)
            pdf.cell(col_w, 8, f"{stats.get('std', 0):.2f}", border=1)
            pdf.cell(col_w, 8, f"{stats.get('min',0):.1f}/{stats.get('max',0):.1f}", border=1)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, "No numeric columns available.", ln=True)

    pdf.add_page()
    
    # Section 3 - ML Model Results
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. ML Model Results", ln=True)
    pdf.set_font("Helvetica", "", 12)
    
    if ml_results.get('best_model_name'):
        pdf.cell(0, 8, f"Task Type: {ml_results.get('task_type', 'N/A').capitalize()}", ln=True)
        pdf.cell(0, 8, f"Best Model: {ml_results.get('best_model_name', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"{ml_results.get('metric_name', 'Metric')}: {ml_results.get('best_metric_value', 'N/A')}", ln=True)
        pdf.ln(5)
        
        # Feature Importance Table
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top 10 Feature Importances", ln=True)
        pdf.ln(2)
        
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(90, 8, "Feature", border=1)
        pdf.cell(50, 8, "Importance Score", border=1)
        pdf.ln()
        
        pdf.set_font("Helvetica", "", 10)
        for feat, score in ml_results.get('feature_importance', {}).items():
            pdf.cell(90, 8, str(feat)[:40], border=1)
            pdf.cell(50, 8, f"{score:.4f}", border=1)
            pdf.ln()
    else:
        pdf.cell(0, 8, "No model trained or evaluated.", ln=True)

    pdf.ln(10)
    
    # Section 4 - NLP Query Transcript
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. NLP Query Transcript", ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    if query_log:
        for entry in query_log:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, f"[{entry['timestamp']}] Q: {entry['question']}", ln=True)
            pdf.set_font("Helvetica", "", 10)
            resp = entry.get('response', {})
            filt = resp.get('filter_code')
            chart = resp.get('chart_type')
            err = resp.get('error')
            
            if err:
                pdf.cell(0, 6, f"Error: {err}", ln=True)
            else:
                pdf.cell(0, 6, f"Filter Code: {filt if filt else 'None'}", ln=True)
                pdf.cell(0, 6, f"Chart Required: {chart if chart else 'None'}", ln=True)
            pdf.ln(3)
    else:
        pdf.cell(0, 8, "No natural language queries logged.", ln=True)

    # fpdf2 outputs directly as bytearray via output() without arguments
    try:
        # return string conceptually, then cast correctly for fpdf2
        return bytes(pdf.output())
    except Exception:
        # Fallback for some fpdf variants
        return pdf.output(dest='S').encode('latin1')
