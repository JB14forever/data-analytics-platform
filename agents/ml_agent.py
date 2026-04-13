# D:\data_analytics_platform\agents\ml_agent.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


class MLAgent:
    """
    Handles automated machine learning model selection, training, and evaluation.
    Now includes an expanded suite of algorithms and comparative leaderboards.
    """

    def detect_task(self, df: pd.DataFrame, target_col: str) -> str:
        target = df[target_col]
        dtype = target.dtype
        n_unique = target.nunique()

        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) or n_unique <= 20:
            return 'classification'
        return 'regression'

    def get_feature_importance(self, model, feature_names: list) -> dict:
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            
        if importances is not None:
            feat_imp = {feat: float(imp) for feat, imp in zip(feature_names, importances)}
            sorted_feats = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True)[:10])
            return sorted_feats
        return {}

    def train(self, df: pd.DataFrame, target_col: str) -> dict:
        """
        Splits data, trains competing tree-based and linear algorithms, 
        and extracts a leaderboard of the results.
        """
        task_type = self.detect_task(df, target_col)
        df_clean = df.dropna(subset=[target_col]).copy()
        
        features = df_clean.drop(columns=[target_col])
        features = features.select_dtypes(include=[np.number, bool])
        
        X = features
        y = df_clean[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        leaderboard = []
        best_model_name = ""
        best_model_obj = None
        best_metric_val = None
        metric_name = ""
        
        if task_type == 'classification':
            metric_name = 'Weighted F1-Score'
            best_metric_val = -1.0
            
            models = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                'Decision Tree': DecisionTreeClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Support Vector Classifier': SVC(random_state=42)
            }
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    leaderboard.append({'Model': name, metric_name: score})
                    
                    if score > best_metric_val:
                        best_metric_val = score
                        best_model_name = name
                        best_model_obj = model
                except Exception:
                    continue
                    
        else:
            metric_name = 'Root Mean Squared Error (RMSE)'
            best_metric_val = float('inf')
            
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': XGBRegressor(random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'Linear Regression': LinearRegression(),
                'Support Vector Regressor': SVR()
            }
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = root_mean_squared_error(y_test, y_pred)
                    
                    leaderboard.append({'Model': name, metric_name: score})
                    
                    if score < best_metric_val:
                        best_metric_val = score
                        best_model_name = name
                        best_model_obj = model
                except Exception:
                    continue
                    
        # Sort Leaderboard natively
        if task_type == 'classification':
            leaderboard = sorted(leaderboard, key=lambda x: x[metric_name], reverse=True)
        else:
            leaderboard = sorted(leaderboard, key=lambda x: x[metric_name], reverse=False)
            
        feature_importance = self.get_feature_importance(best_model_obj, X.columns.tolist())
        
        return {
            'task_type': task_type,
            'best_model_name': best_model_name,
            'best_model_obj': best_model_obj,
            'best_metric_value': round(float(best_metric_val), 4) if best_metric_val not in (-1.0, float('inf')) else 0.0,
            'metric_name': metric_name,
            'feature_importance': feature_importance,
            'leaderboard': leaderboard
        }
