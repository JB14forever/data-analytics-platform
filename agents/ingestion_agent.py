# D:\data_analytics_platform\agents\ingestion_agent.py

import io
import csv
import pandas as pd


class IngestionAgent:
    """
    Handles the ingestion of raw data and initial profiling.
    
    This agent follows the Single Responsibility Principle by only focusing
    on reading data safely into memory and deriving fundamental schema
    and quality statistics.
    """

    def load_csv(self, file) -> pd.DataFrame:
        """
        Loads an uploaded CSV file from Streamlit into a DataFrame in-memory.
        
        Uses Python's csv.Sniffer to automatically detect the delimiter so the
        user does not have to specify whether the file is comma, tab, or semi-colon
        separated. The file is never written to disk.
        
        Args:
            file: Streamlit UploadedFile object containing the CSV bytes.
            
        Returns:
            pd.DataFrame: The loaded raw pandas DataFrame.
            
        Design Decision:
            We use io.StringIO and python's csv.Sniffer rather than pandas' default
            C-engine sniffer for more robust delimiter detection on messy files.
        """
        # Read the file content as strings
        content = file.getvalue().decode('utf-8')
        
        # Sniff delimiter using a sample of the first 2048 characters
        sample = content[:2048]
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            # Fallback if sniffing fails
            delimiter = ','
            
        # Re-wrap string into a file-like object for pandas
        str_io = io.StringIO(content)
        df = pd.read_csv(str_io, sep=delimiter)
        return df

    def infer_schema(self, df: pd.DataFrame) -> dict:
        """
        Infers the schema and profiling metrics for all columns in the DataFrame.
        
        Args:
            df (pd.DataFrame): The raw data.
            
        Returns:
            dict: A mapping from column name to its profiling details:
                - dtype (str)
                - null_count (int)
                - null_percent (float)
                - cardinality (int)
        """
        schema = {}
        total_rows = len(df)
        
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            null_pct = (null_count / total_rows) * 100 if total_rows > 0 else 0
            
            # Simple column dtype mapping
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
        """
        Computes an overall data health score out of 100 based on quality penalties.
        
        Weighting:
            - Null Percentage: 0.5 w
            - Duplicate Row Percentage: 0.3 w
            - High-dominance Column Percentage: 0.2 w
            
        Args:
            df (pd.DataFrame): The raw dataset.
            
        Returns:
            float: Score clamped between 0 and 100.
            
        Mathematical Rationale:
            Penalty = (overall_null_pct * 0.5) + (dup_row_pct * 0.3) + (dominant_cols_pct * 0.2)
            Score = max(0, 100 - Penalty)
            This weighted formula heavily penalizes missing values, moderately penalizes
            duplicate information, and slightly penalizes uninformative features.
        """
        if df.empty:
            return 0.0

        total_cells = df.shape[0] * df.shape[1]
        overall_null_pct = (df.isnull().sum().sum() / total_cells) * 100
        
        dup_row_pct = (df.duplicated().sum() / len(df)) * 100
        
        # Calculate columns with >95% single-value dominance
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

    def flag_quality_issues(self, df: pd.DataFrame) -> list[str]:
        """
        Scans DataFrame and returns human-readable warnings for common data issues.
        
        Args:
            df (pd.DataFrame): Dataset to scan.
            
        Returns:
            list[str]: Collection of warning strings describing the anomalies.
        """
        issues = []
        total_rows = len(df)
        
        if total_rows == 0:
            return ["Dataset is empty."]
            
        for col in df.columns:
            # Nulls check
            null_pct = (df[col].isnull().sum() / total_rows) * 100
            if null_pct > 30:
                issues.append(f"Column '{col}' has {null_pct:.1f}% missing values (High sparsity).")
            
            # Identical values check (cardinality == 1 ignoring nulls)
            if df[col].nunique() == 1:
                val = df[col].dropna().iloc[0]
                issues.append(f"Column '{col}' has all identical values ('{val}'). It provides no variance.")
                
            # Datetime masquerading as object check
            if df[col].dtype == 'object':
                # Try to parse a small non-null sample to see if it feels like datetime
                sample = df[col].dropna().head(10)
                if not sample.empty:
                    # simplistic check: if it can be coerced cleanly, flag it
                    try:
                        pd.to_datetime(sample, errors='raise')
                        issues.append(f"Column '{col}' contains datetime strings but is stored as object dtype.")
                    except (ValueError, TypeError):
                        pass

        return issues
