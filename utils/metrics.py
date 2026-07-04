import numpy as np
import streamlit as st
from utils.preprocessing import preprocess_df
from utils.loaders import load_models, algonames

def calculate_model_metrics(df):
    """Calculates classification metrics dynamically for all loaded models on the dataset."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    X = preprocess_df(df).values.astype(float)
    y = df['HeartDisease'].values
    
    models = load_models()
    metrics = {}
    
    for name in algonames:
        model = models.get(name)
        if model is None:
            metrics[name] = None
            continue
            
        try:
            preds = model.predict(X)
            
            # Extract probability or decision value for ROC-AUC
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
            elif hasattr(model, 'decision_function'):
                probs = model.decision_function(X)
            else:
                probs = preds
                
            metrics[name] = {
                'accuracy': accuracy_score(y, preds),
                'precision': precision_score(y, preds),
                'recall': recall_score(y, preds),
                'f1_score': f1_score(y, preds),
                'roc_auc': roc_auc_score(y, probs)
            }
        except Exception as e:
            st.error(f"❌ Error calculating metrics for {name}: {e}")
            metrics[name] = None
            
    return metrics
