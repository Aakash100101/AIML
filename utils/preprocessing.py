import pandas as pd

def preprocess_df(df):
    """Encodes categorical columns in heart.csv to match model input conventions."""
    encoded_df = df.copy()
    
    # Categorical variable mappings matching Tab 1 encoding
    encoded_df['Sex'] = encoded_df['Sex'].map({'M': 0, 'F': 1})
    encoded_df['ChestPainType'] = encoded_df['ChestPainType'].map({'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3})
    encoded_df['RestingECG'] = encoded_df['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    encoded_df['ExerciseAngina'] = encoded_df['ExerciseAngina'].map({'N': 0, 'Y': 1})
    encoded_df['ST_Slope'] = encoded_df['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})
    
    # Rename columns to match input_data features in app.py
    rename_dict = {
        'Age': 'age',
        'Sex': 'sex',
        'ChestPainType': 'chest_pain_type',
        'RestingBP': 'resting_bp',
        'Cholesterol': 'cholesterol',
        'FastingBS': 'fasting_blood_sugar',
        'RestingECG': 'resting_ecg',
        'MaxHR': 'max_heart_rate',
        'ExerciseAngina': 'exercise_angina',
        'Oldpeak': 'oldpeak',
        'ST_Slope': 'st_slope'
    }
    encoded_df = encoded_df.rename(columns=rename_dict)
    
    feature_cols = [
        'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
        'exercise_angina', 'oldpeak', 'st_slope'
    ]
    return encoded_df[feature_cols]
