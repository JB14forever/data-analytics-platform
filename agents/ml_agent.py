# D:\data_analytics_platform\agents\ml_agent.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor


class MLAgent:
    """
    Handles automated machine learning model selection, training, and evaluation.
    """

    def detect_task(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Determines whether the target variable requires a classification or regression model.
        
        Args:
            df (pd.DataFrame): The dataset.
            target_col (str): The column name to predict.
            
        Returns:
            str: 'classification' or 'regression'.
            
        Design Decision:
            If the target column is clearly an object, a boolean, or has twenty or fewer 
            unique discrete values, we treat it as a classification task (categorical). 
            Otherwise, we assume a continuous regression distribution. 
            20 is used as a heuristic threshold for ordinal/discrete multi-class scenarios.
        """
        target = df[target_col]
        dtype = target.dtype
        n_unique = target.nunique()

        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or n_unique <= 20:
            return 'classification'
        return 'regression'

    def get_feature_importance(self, model, feature_names: list) -> dict:
        """
        Extracts and sorts the top 10 feature importances from a tree-based model.
        
        Args:
            model: Trained sklearn or xgboost model.
            feature_names (list): List of feature column names.
            
        Returns:
            dict: Top 10 features mapped to their importance score (descending order).
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
            # Sort by descending importance, take top 10
            sorted_feats = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:10])
            return sorted_feats
        return {}

    def train(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Splits data, trains competing tree-based algorithms, and selects the best model.
        
        Args:
            df (pd.DataFrame): Transformed dataset containing target and features.
            target_col (str): Column to predict.
            
        Returns:
            dict: ML results including the best model and relevant metrics.
            
        Mathematical Rationale (Metrics selection):
            - **Classification (Weighted F1-Score)**: F1 perfectly balances Precision 
              and Recall. We use 'weighted' to account for imbalanced class distributions 
              so majority classes don't heavily distort the perceived model accuracy.
            - **Regression (RMSE)**: Root Mean Squared Error heavily penalizes larger 
              variances (errors) between the predicted and actual values. Since the errors 
              are squared before being averaged, large outliers affect the score significantly, 
              giving a stricter representation of model fitness compared to MAE.
        """
        task_type = self.detect_task(df, target_col)
        
        # Ensure target is clean
        df_clean = df.dropna(subset=[target_col]).copy()
        
        # In transformed data, drop explicit datetime or object columns untouched
        features = df_clean.drop(columns=[target_col])
        features = features.select_dtypes(include=[np.number, bool])
        
        X = features
        y = df_clean[target_col]
        
        # Basic 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        best_model_name = ""
        best_model_obj = None
        best_metric_val = None
        metric_name = ""
        
        if task_type == 'classification':
            metric_name = 'Weighted F1-Score'
            best_metric_val = -1.0
            
            # Map labels to 0-N integers for XGBoost if needed
            y_train_mapped = y_train
            y_test_mapped = y_test
            # Using basic RF and XGB
            models = {
                'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBoost Classifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            }
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train_mapped)
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
                    if score > best_metric_val:
                        best_metric_val = score
                        best_model_name = name
                        best_model_obj = model
                except Exception:
                    # Depending on data structure, XGBoost might fail without proper encoding. Fall back gracefully.
                    continue
                    
        else:
            metric_name = 'Root Mean Squared Error'
            best_metric_val = float('inf')
            
            models = {
                'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost Regressor': XGBRegressor(random_state=42)
            }
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # RMSE calculation
                score = root_mean_squared_error(y_test, y_pred)
                if score < best_metric_val:
                    best_metric_val = score
                    best_model_name = name
                    best_model_obj = model
                    
        # Feature Importance Extraction
        feature_importance = self.get_feature_importance(best_model_obj, X.columns.tolist())
        
        return {
            'task_type': task_type,
            'best_model_name': best_model_name,
            'best_model_obj': best_model_obj,
            'best_metric_value': round(float(best_metric_val), 4) if best_metric_val not in (-1.0, float('inf')) else 0.0,
            'metric_name': metric_name,
            'feature_importance': feature_importance
        }
