# D:\data_analytics_platform\agents\cleaning_agent.py

import pandas as pd
import numpy as np
import re
from scipy.stats import skew


class CleaningAgent:
    """
    Cleans the dataset utilizing the rigorous 12-Step Architecture:
    1. Understand limits, 2. Handle missing, 3. Duplicate removal
    4. Type fixing, 5. Standardization, 6. Outliers
    7. Text cleaning, 8. Inconsistency, 9. Validation.
    """

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 5: Standardize Column Names"""
        df_clean = df.copy()
        new_cols = []
        for col in df_clean.columns:
            # Lowercase, replace spaces with _, remove non-alphanumeric (except _)
            c = str(col).lower().strip()
            c = c.replace(' ', '_')
            c = re.sub(r'[^a-z0-9_]', '', c)
            new_cols.append(c)
        df_clean.columns = new_cols
        return df_clean

    def fix_data_types_and_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 4 & 7: Convert to correct types and clean text strings."""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Try converting to datetime first
                sample = df_clean[col].dropna().head(20).astype(str)
                # Quick heuristic to avoid casting pure text to datetime arbitrarily
                if sample.str.match(r'^\d{4}-\d{2}-\d{2}|^\d{2}/\d{2}/\d{4}').any():
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
                    except Exception:
                        pass
                
                # If it's still object, clean the text
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
                    # Step 8 handle inconsistencies like "male " -> "male", already largely fixed by strip/lower.
        return df_clean

    def handle_missing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Step 2: Missing Values
        >30% -> drop
        <30% -> Mean/Median/Mode impute
        """
        df_clean = df.copy()
        total_rows = len(df_clean)
        dropped_cols = {}
        
        for col in df_clean.columns:
            null_count = df_clean[col].isnull().sum()
            if null_count == 0:
                continue
                
            null_pct = (null_count / total_rows) * 100
            
            if null_pct > 30:
                df_clean = df_clean.drop(columns=[col])
                dropped_cols[col] = f"Dropped due to severe missingness (>30% missing: {null_pct:.1f}%)."
            else:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    c_skew = skew(df_clean[col].dropna())
                    if pd.isna(c_skew) or abs(c_skew) > 0.5:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    else:
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].ffill().bfill()
                else:
                    mode_s = df_clean[col].mode()
                    if not mode_s.empty:
                        df_clean[col] = df_clean[col].fillna(mode_s[0])
                        
        return df_clean, dropped_cols

    def handle_outliers_and_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6 & 9: Winsorize outliers and validate ranges.
        """
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Range validation: absolute heuristics
            # If standard deviations are incredibly tight around positives, clip negatives
            import math
            if 'age' in col or 'salary' in col or 'price' in col or 'amount' in col:
                df_out[col] = df_out[col].clip(lower=0)
                
            # Outlier Handling via IQR
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])
            
        return df_out

    def remove_duplicates(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Step 3: Remove Duplicate Rows"""
        initial_len = len(df)
        df_dedup = df.drop_duplicates().copy()
        removed = initial_len - len(df_dedup)
        return df_dedup, removed

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, int]:
        """
        Orchestrates the 12-Step pipeline up to step 9. 
        Encoding and Scaling are delegated to TransformationAgent.
        """
        # Step 5: Standardization
        df_step = self.standardize_columns(df)
        
        # Step 4, 7, 8: Type fixing and Text standardization
        df_step = self.fix_data_types_and_text(df_step)
        
        # Step 3: Duplicate removal
        df_step, duplicates_removed = self.remove_duplicates(df_step)
        
        # Step 2: Missing Values
        df_step, missing_drops = self.handle_missing(df_step)
        
        # Step 6 & 9: Outliers & Range Validation
        df_step = self.handle_outliers_and_ranges(df_step)
        
        return df_step, missing_drops, duplicates_removed
