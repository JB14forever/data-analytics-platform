# D:\data_analytics_platform\agents\transformation_agent.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TransformationAgent:
    """
    Transforms clean data into a format optimal for Machine Learning.
    Includes feature scaling and categorical encoding.
    """

    def encode(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Encodes categorical and boolean variables into numerical formats.
        
        Args:
            df (pd.DataFrame): Cleaned dataset.
            schema (dict): Schema dictionary for identifying column contexts.
            
        Returns:
            pd.DataFrame: Dataset with encoded nominal/categorical features.
            
        Design Decision:
            - **Low Cardinality (<=10)**: Uses One-Hot Encoding (pd.get_dummies).
              Creating a column per category creates independence but expands dimensions.
            - **High Cardinality (>10)**: Uses Label Encoding. We avoid OHE for highly 
              variant categoricals to prevent the "Curse of Dimensionality" (feature space 
              explosion leading to sparsity and model overfitting).
        """
        df_encoded = df.copy()
        
        for col, meta in schema.items():
            if meta['dtype'] == 'categorical':
                if meta['cardinality'] <= 10:
                    # One Hot Encoding
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
                else:
                    # Label Encoding
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    
            elif meta['dtype'] == 'boolean':
                # Map booleans to integers
                df_encoded[col] = df_encoded[col].astype(int)
                
        return df_encoded

    def scale(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Normalizes numeric columns to have a mean of 0 and std deviation of 1.
        
        Args:
            df (pd.DataFrame): Encoded dataset.
            schema (dict): Original schema dict.
            
        Returns:
            pd.DataFrame: Dataset with scaled numeric columns.
            
        Mathematical Rationale (Z-Score Normalization):
            Z = (X - μ) / σ
            Where μ is the mean and σ is the standard deviation.
            Scaling prevents variables with inherently larger magnitudes (e.g., salary vs age) 
            from dominating variance calculations or gradient descents in ML loss functions.
        """
        df_scaled = df.copy()
        
        # Identify columns that are purely numeric (this includes original numeric and label encoded)
        # OHE columns will be uint8 or bool in pandas, which is generally fine to scale or ignore.
        # We will scale purely numeric continuous variables originally indicated in schema.
        numeric_cols = [c for c in df_scaled.columns if c in schema and schema[c]['dtype'] == 'numeric']
        
        if numeric_cols:
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            
        return df_scaled

    def transform(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Orchestrates transformation sequence.
        
        Sequence:
        1. Encode categorical/boolean features to numerical representations.
        2. Scale numeric columns for un-biased magnitude modeling.
        
        Args:
            df (pd.DataFrame): Cleaned DataFrame.
            schema (dict): Schema metadata.
            
        Returns:
            pd.DataFrame: Fully formatted model-ready DataFrame.
        """
        df_encoded = self.encode(df, schema)
        df_scaled = self.scale(df_encoded, schema)
        return df_scaled
