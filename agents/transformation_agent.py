# D:\data_analytics_platform\agents\transformation_agent.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class TransformationAgent:
    """
    Transforms clean data into a format optimal for Machine Learning.
    Step 10 & 11: Encoding and Feature Scaling.
    """

    def handle_datetime_features(self, df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list]:
        """
        Extracts temporal features from datetimes and drops the original fields
        since ML models cannot digest raw datetime64 objects directly.
        """
        df_feat = df.copy()
        dt_cols = [c for c in schema if schema[c]['dtype'] == 'datetime']
        dropped = []
        
        for col in dt_cols:
            if col in df_feat.columns:
                df_feat[f"{col}_year"] = df_feat[col].dt.year
                df_feat[f"{col}_month"] = df_feat[col].dt.month
                df_feat[f"{col}_day"] = df_feat[col].dt.day
                # After extracting numerics, toss original
                df_feat = df_feat.drop(columns=[col])
                dropped.append(col)
                
        return df_feat, dropped

    def encode(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        df_encoded = df.copy()
        
        for col, meta in schema.items():
            if col not in df_encoded.columns:
                continue
                
            if meta['dtype'] == 'categorical':
                if meta['cardinality'] <= 10:
                    df_encoded = pd.get_dummies(df_encoded, columns=[col], drop_first=True)
                else:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    
            elif meta['dtype'] == 'boolean':
                df_encoded[col] = df_encoded[col].astype(int)
                
        return df_encoded

    def scale(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        df_scaled = df.copy()
        
        # We only strictly scale original base numeric variables. 
        # OHE and encoded labels often shouldn't be scaled, but StandardScaling them 
        # isn't universally harmful for Trees/LinReg. To be precise, scale all purely continous:
        numeric_cols = [c for c in df_scaled.columns if c in schema and schema[c]['dtype'] == 'numeric']
        
        if numeric_cols:
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            
        return df_scaled

    def transform(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        Orchestrates step 10 & 11 transformation sequence.
        """
        # 1. Parse Datetimes to int features
        df_dt, dt_drops = self.handle_datetime_features(df, schema)
        
        # 2. Encode
        df_encoded = self.encode(df_dt, schema)
        
        # 3. Scale
        df_scaled = self.scale(df_encoded, schema)
        
        return df_scaled
