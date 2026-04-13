# D:\data_analytics_platform\agents\ingestion_agent.py

import io
import csv
import pandas as pd


class IngestionAgent:
    """
    Handles the ingestion of raw data (CSV and Excel) and initial profiling.
    Detects and filters out useless primary keys and zero-variance columns.
    """

    def load_data(self, file) -> pd.DataFrame:
        """
        Loads an uploaded file (CSV or Excel) into a DataFrame.
        """
        filename = file.name.lower()
        
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            # openpyxl handles xlsx, xlrd handles xls. We assume xlsx here.
            df = pd.read_excel(file.getvalue())
            return df
        else:
            # Fallback to smart CSV sniffing
            content = file.getvalue().decode('utf-8', errors='replace')
            sample = content[:2048]
            try:
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                delimiter = ','
                
            str_io = io.StringIO(content)
            df = pd.read_csv(str_io, sep=delimiter)
            return df

    def filter_primary_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Removes identifiers, 100% unique string columns, and zero-variance columns.
        
        Returns:
            df (pd.DataFrame): Filtered DataFrame
            dropped_reasoning (dict): A log of what was dropped and why.
        """
        df_filtered = df.copy()
        dropped = {}
        total_rows = len(df_filtered)
        
        if total_rows == 0:
            return df_filtered, dropped
            
        cols_to_drop = []
        
        for col in df_filtered.columns:
            # 1. Zero-Variance
            if df_filtered[col].nunique(dropna=False) <= 1:
                dropped[col] = "Zero-variance (All values are identical or completely missing)."
                cols_to_drop.append(col)
                continue
                
            # 2. Heuristic ID naming
            lname = str(col).lower()
            if lname in ['id', 'uuid', 'index'] or lname.endswith('_id') or lname.endswith(' id'):
                # Only drop if it actually looks like an ID (high cardinality)
                if df_filtered[col].nunique() > (total_rows * 0.5):
                    dropped[col] = f"Detected as a primary key/ID by name ('{col}')."
                    cols_to_drop.append(col)
                    continue
                    
            # 3. 100% Unique Strings (e.g. emails, usernames, hashes)
            if df_filtered[col].dtype == 'object':
                if df_filtered[col].nunique() == total_rows:
                    dropped[col] = "100% unique string values (Likely a primary key or arbitrary identifier)."
                    cols_to_drop.append(col)

        if cols_to_drop:
            df_filtered = df_filtered.drop(columns=cols_to_drop)
            
        return df_filtered, dropped

    def infer_schema(self, df: pd.DataFrame) -> dict:
        """
        Infers the schema and profiling metrics for all columns.
        """
        schema = {}
        total_rows = len(df)
        
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_type = 'datetime'
            elif pd.api.types.is_bool_dtype(df[col]):
                col_type = 'boolean'
            else:
                col_type = 'categorical'

            schema[col] = {
                'dtype': col_type,
                'null_count': null_count,
                'null_percentage': null_pct,
                'cardinality': int(df[col].nunique(dropna=False))
            }
            
        return schema

    def compute_health_score(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0

        total_cells = df.shape[0] * df.shape[1]
        overall_null_pct = (df.isnull().sum().sum() / total_cells) * 100
        dup_row_pct = (df.duplicated().sum() / len(df)) * 100
        
        dominant_cols = 0
        for col in df.columns:
            if df[col].nunique() > 0:
                top_val_freq = df[col].value_counts(normalize=True).iloc[0] * 100
                if top_val_freq > 95:
                    dominant_cols += 1
        
        dominant_cols_pct = (dominant_cols / df.shape[1]) * 100
        
        penalty = (overall_null_pct * 0.5) + (dup_row_pct * 0.3) + (dominant_cols_pct * 0.2)
        score = max(0.0, min(100.0, 100.0 - penalty))
        
        return round(score, 2)
