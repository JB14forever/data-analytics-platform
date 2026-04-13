# D:\data_analytics_platform\agents\cleaning_agent.py

import pandas as pd
import numpy as np
from scipy.stats import skew


class CleaningAgent:
    """
    Cleans the parsed dataset by handling missing values, duplicates, and outliers.
    """

    def impute(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Handles missing values (NaNs) across the DataFrame.
        
        Args:
            df (pd.DataFrame): The dataset (modified out-of-place).
            schema (dict): Schema dictionary output from IngestionAgent.
            
        Returns:
            pd.DataFrame: A new DataFrame with filled missing values.
            
        Design Decision / Mathematical Rationale:
            - **Numeric Columns**: Checks the distribution's skewness. 
              If the absolute skewness > 0.5, the data is asymmetric and the median 
              is used for imputation to maintain robustness against heavy tails. 
              Otherwise, the mean is used since symmetric distributions center nicely on it.
            - **Categorical Columns**: The mode (most frequent value) is used to maintain
              the dominant category distribution.
        """
        df_clean = df.copy()
        
        for col, meta in schema.items():
            if df_clean[col].isnull().sum() == 0:
                continue
                
            if meta['dtype'] == 'numeric':
                # compute skew on non-nulls only
                col_skew = skew(df_clean[col].dropna())
                # Scipy returns NaN or masked array for constant input sizes or 0 variance
                if pd.isna(col_skew) or abs(col_skew) > 0.5:
                    fill_val = df_clean[col].median()
                else:
                    fill_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(fill_val)
                
            elif meta['dtype'] == 'categorical' or meta['dtype'] == 'boolean':
                mode_serie = df_clean[col].mode()
                if not mode_serie.empty:
                    fill_val = mode_serie[0]
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    
            elif meta['dtype'] == 'datetime':
                # Datetime is forward/backward filled or left alone. We'll use ffill for simplicity.
                df_clean[col] = df_clean[col].ffill().bfill()
                
        return df_clean

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes identical duplicate rows from the dataset.
        
        Args:
            df (pd.DataFrame): The dataset representing the latest state.
            
        Returns:
            pd.DataFrame: Deduplicated DataFrame.
        """
        initial_len = len(df)
        df_dedup = df.drop_duplicates().copy()
        removed = initial_len - len(df_dedup)
        # We could log this if standard logging is implemented. We keep it silent here 
        # and just return the structured df. The UI handles reporting general health.
        return df_dedup

    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Limits extreme values in numeric columns using Winsorization.
        
        Args:
            df (pd.DataFrame): The dataset.
            method (str): Method for outlier capping. Only 'iqr' is supported.
            
        Returns:
            pd.DataFrame: The bounded DataFrame.
            
        Mathematical Rationale:
            IQR (Inter-Quartile Range) = Q3 (75th percentile) - Q1 (25th percentile).
            Lower Bound = Q1 - (1.5 * IQR)
            Upper Bound = Q3 + (1.5 * IQR)
            Values extending beyond these boundaries pull averages and linear models 
            inaccurately. Winsorizing (capping instead of dropping) ensures we don't 
            lose viable rows while nullifying extreme magnitudes.
        """
        if method != 'iqr':
            raise ValueError(f"Unknown outlier handling method: {method}")
            
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q1 = df_out[col].quantile(0.25)
            q3 = df_out[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Application of bounds (Winsorizing)
            df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
            df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])
            
        return df_out

    def clean(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Orchestrates the entire cleaning pipeline sequentially.
        
        Sequence:
        1. Impute missing values.
        2. Remove exact duplicate rows.
        3. Cap outliers using IQR.
        
        Args:
            df (pd.DataFrame): Raw DataFrame.
            schema (dict): Profiling data dictionary.
            
        Returns:
            pd.DataFrame: Fully cleaned DataFrame.
        """
        df_clean = self.impute(df, schema)
        df_dedup = self.remove_duplicates(df_clean)
        df_final = self.handle_outliers(df_dedup, method='iqr')
        
        return df_final
