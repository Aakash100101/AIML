import streamlit as st
import pandas as pd
import pickle
import os

# Resolve paths relative to the project root directory
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

modelnames = [
    os.path.join(MODEL_DIR, 'models', 'DecisionTreeClassifier.pkl'),
    os.path.join(MODEL_DIR, 'models', 'LogisticRegression.pkl'),
    os.path.join(MODEL_DIR, 'models', 'RandomForestClassifier.pkl'),
    os.path.join(MODEL_DIR, 'models', 'SVC.pkl')
]

algonames = [
    'Decision Trees',
    'Logistic Regression',
    'Random Forest',
    'Support Vector Machine'
]

@st.cache_data
def load_dataset():
    """Loads and caches the heart disease dataset."""
    csv_path = os.path.join(MODEL_DIR, 'data', 'heart.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)

@st.cache_resource
def load_models():
    """Loads and caches all four models. Returns a dictionary mapping model names to objects."""
    models = {}
    for name, filepath in zip(algonames, modelnames):
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.error(f"❌ Error loading model '{name}' from {filepath}: {e}")
            models[name] = None
    return models
