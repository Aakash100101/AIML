
import pickle
import pandas as pd
import streamlit as st
import numpy as np
import os

# -------------------- PAGE SETUP -------------------- #
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# -------------------- ANIMATED CSS -------------------- #
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global font */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Animated gradient background for title */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Pulsing heart animation */
    @keyframes heartbeat {
        0%, 100% { transform: scale(1); }
        10%, 30% { transform: scale(0.9); }
        20%, 40% { transform: scale(1.1); }
    }
    
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Slide in from left */
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Slide in from right */
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Glowing effect */
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(255, 75, 75, 0.5); }
        50% { box-shadow: 0 0 20px rgba(255, 75, 75, 0.8), 0 0 30px rgba(255, 75, 75, 0.6); }
    }
    
    /* Floating animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Apply animations to elements */
    .stApp > header {
        animation: fadeIn 1s ease-in;
    }
    
    h1 {
        animation: fadeIn 1.2s ease-in;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        animation: slideInLeft 0.8s ease-out;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        padding: 12px 24px;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        animation: fadeIn 0.6s ease-in;
    }
    
    /* Input field animations and styling */
    .stNumberInput, .stSelectbox {
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Enhanced input styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Button hover effects */
    .stButton > button {
        transition: all 0.3s ease;
        animation: fadeIn 1s ease-in;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 12px 32px !important;
        font-size: 16px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Success/Error message animations */
    .stSuccess, .stError, .stWarning {
        animation: slideInRight 0.5s ease-out;
        border-radius: 10px;
    }
    
    /* Expander animation */
    .streamlit-expanderHeader {
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: rgba(102, 126, 234, 0.1);
        transform: translateX(5px);
    }
    
    /* Card-like container with shadow */
    .element-container {
        animation: fadeIn 0.7s ease-in;
    }
    
    /* Metric animation */
    [data-testid="stMetricValue"] {
        animation: fadeIn 1s ease-in;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("❤ Heart Disease Predictor")

# -------------------- TABS -------------------- #
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

# -------------------- MODEL NAMES & PATH -------------------- #
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
modelnames = [
    os.path.join(MODEL_DIR, 'DecisionTreeClassifier.pkl'),
    os.path.join(MODEL_DIR, 'LogisticRegression.pkl'),
    os.path.join(MODEL_DIR, 'RandomForestClassifier.pkl'),
    os.path.join(MODEL_DIR, 'SVC.pkl')
]
algonames = [
    'Decision Trees',
    'Logistic Regression',
    'Random Forest',
    'Support Vector Machine'
]

# -------------------- TAB 1: SINGLE PREDICTION -------------------- #
with tab1:
    # Hero Section with Animated Header
    st.markdown("""
    <style>
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 30px;
            animation: fadeIn 0.8s ease-in;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        .hero-section::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: float 6s ease-in-out infinite;
        }
        .hero-section h2 {
            color: white;
            margin: 0;
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            animation: slideInLeft 0.8s ease-out;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            position: relative;
            z-index: 1;
        }
        .hero-section p {
            color: rgba(255,255,255,0.9);
            text-align: center;
            margin-top: 10px;
            font-size: 16px;
            position: relative;
            z-index: 1;
        }
        
        .input-card {
            background: white;
            padding: 30px;
            border-radius: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            animation: fadeIn 1s ease-in;
            border: 1px solid rgba(102, 126, 234, 0.1);
            transition: all 0.3s ease;
        }
        
        .input-card:hover {
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.12);
            transform: translateY(-2px);
        }
        
        .section-divider {
            display: flex;
            align-items: center;
            margin: 25px 0;
        }
        
        .section-divider::before,
        .section-divider::after {
            content: '';
            flex: 1;
            height: 2px;
            background: linear-gradient(to right, transparent, #667eea, transparent);
        }
        
        .section-title {
            color: #667eea;
            font-weight: 600;
            font-size: 18px;
            margin: 0 15px;
            white-space: nowrap;
        }
        
        .info-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            display: inline-block;
            margin: 10px 5px;
            animation: fadeIn 1.2s ease-in;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
    </style>
    <div class="hero-section">
        <h2>🩺 Patient Health Assessment</h2>
        <p>Enter patient details below for comprehensive heart disease risk analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info badges
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span class="info-badge">✓ 4 ML Models</span>
        <span class="info-badge">✓ 85% Accuracy</span>
        <span class="info-badge">✓ Instant Results</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Personal Information Card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"><span class="section-title">👤 Personal Information</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("👶 Age (years)", min_value=0, max_value=150, help="Patient's age in years")
        sex = st.selectbox("⚧ Sex", ["Male", "Female"], help="Biological sex of the patient")
        
    with col2:
        resting_bp = st.number_input("💓 Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, help="Blood pressure at rest")
        cholesterol = st.number_input("🩸 Serum Cholesterol (mg/dl)", min_value=0, help="Cholesterol level in blood")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clinical Measurements Card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"><span class="section-title">🏥 Clinical Measurements</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        chest_pain = st.selectbox(
            "💢 Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            help="Type of chest pain experienced"
        )
        fasting_bs = st.selectbox("🍬 Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"], help="Blood sugar after fasting")
        resting_ecg = st.selectbox(
            "📊 Resting ECG Results",
            ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
            help="Electrocardiogram results"
        )
    with col2:
        max_hr = st.number_input("💗 Maximum Heart Rate", min_value=60, max_value=202, help="Highest heart rate during exercise")
        exercise_angina = st.selectbox("🏃 Exercise-Induced Angina", ["No", "Yes"], help="Chest pain during exercise")
        oldpeak = st.number_input("📉 Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, help="ST depression value")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional Tests Card
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"><span class="section-title">🔬 Additional Tests</span></div>', unsafe_allow_html=True)
    
    st_slope = st.selectbox(
        "📈 Slope of Peak Exercise ST Segment", 
        ["Upsloping", "Flat", "Downsloping"],
        help="Slope pattern during exercise test"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # -------------------- ENCODING -------------------- #
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # -------------------- INPUT DATAFRAME -------------------- #
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'chest_pain_type': [chest_pain],
        'resting_bp': [resting_bp],
        'cholesterol': [cholesterol],
        'fasting_blood_sugar': [fasting_bs],
        'resting_ecg': [resting_ecg],
        'max_heart_rate': [max_hr],
        'exercise_angina': [exercise_angina],
        'oldpeak': [oldpeak],
        'st_slope': [st_slope]
    })

    # -------------------- PREDICTION FUNCTION -------------------- #
    def predict_heart_disease(dataframe):
        """Make predictions using all models - handles single and multiple rows."""
        predictions = []
        data = dataframe.values.astype(float)

        for modelname in modelnames:
            try:
                if not os.path.exists(modelname):
                    raise FileNotFoundError(f"Model file not found: {modelname}")
                
                model = pickle.load(open(modelname, 'rb'))
                if hasattr(model, "n_features_in_") and model.n_features_in_ != data.shape[1]:
                    raise ValueError(
                        f"Model expects {model.n_features_in_} features, but got {data.shape[1]}"
                    )
                
                # Predict for all rows at once
                prediction = model.predict(data)
                predictions.append(prediction)
                    
            except Exception as e:
                st.error(f"❌ Error with {os.path.basename(modelname)}: {e}")
                predictions.append(None)
        
        return predictions

    # -------------------- BUTTON & RESULTS -------------------- #
    st.markdown("""
    <style>
        .submit-section {
            text-align: center;
            margin: 30px 0;
            animation: fadeIn 1.2s ease-in;
        }
    </style>
    <div class="submit-section">
    """, unsafe_allow_html=True)
    
    submit_button = st.button("🔍 Analyze Heart Disease Risk", use_container_width=False, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_button:
        # Animated results header
        st.markdown("""
        <style>
            .results-container {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 25px;
                border-radius: 20px;
                margin: 30px 0;
                animation: slideInRight 0.6s ease-out;
                box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
            }
            .results-container h3 {
                color: white;
                margin: 0 0 10px 0;
                text-align: center;
                font-size: 28px;
                font-weight: 700;
                animation: heartbeat 1.5s infinite;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            }
            .model-result-card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                margin: 15px 0;
                animation: fadeIn 0.8s ease-in;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                transition: all 0.3s ease;
                border-left: 5px solid #667eea;
            }
            .model-result-card:hover {
                transform: translateX(8px);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
            }
            .overall-results-box {
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 30px;
                border-radius: 20px;
                margin: 30px 0;
                animation: fadeIn 1s ease-in;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                border: 2px solid rgba(102, 126, 234, 0.2);
            }
            .prediction-badge {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 14px;
                animation: fadeIn 0.6s ease-in;
            }
            .badge-positive {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
                color: white;
            }
            .badge-negative {
                background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
                color: white;
            }
        </style>
        <div class="results-container">
            <h3>🩺 Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)

        results = predict_heart_disease(input_data)

        # Display individual model results in styled cards
        for algo, result in zip(algonames, results):
            st.markdown('<div class="model-result-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.markdown(f"**🤖 {algo}**")
            with col2:
                st.write("Prediction:")
            with col3:
                if result is None:
                    st.warning("⚠ Model error")
                elif result[0] == 1:
                    st.markdown('<span class="prediction-badge badge-positive">❤️ Heart Disease</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="prediction-badge badge-negative">✅ Healthy</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        valid_results = [r[0] for r in results if r is not None]
        if valid_results:
            st.markdown('<div class="overall-results-box">', unsafe_allow_html=True)
            
            # Header with emoji based on result
            positive_count = sum(valid_results)
            total_models = len(valid_results)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🤖 Models Analyzed", total_models)
            with col2:
                st.metric("✅ Agreement", f"{positive_count}/{total_models}")
            with col3:
                agreement_pct = (max(positive_count, total_models - positive_count) / total_models) * 100
                st.metric("📊 Consensus", f"{agreement_pct:.0f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)

            if positive_count > total_models / 2:
                st.error(f"🚨 *Heart Disease Detected* ({positive_count}/{total_models} models agree)")
                with st.expander("💓 View Personalized Health Tips"):
                    st.markdown("""
                    ### ❤️ **Heart Disease Prevention & Care Tips**
                    - 🥗 Eat Heart-Healthy Foods
                    - 🚶 Walk 30 mins every day
                    - 🚭 Quit smoking & alcohol
                    - 🧘 Reduce stress  
                    - 💊 Follow prescriptions  
                    """)
                    st.warning("📞 Contact your doctor immediately if symptoms appear.")

            elif positive_count == total_models / 2:
                st.warning(f"⚠ *Inconclusive Result* ({positive_count}/{total_models} models indicate heart disease)")
                with st.expander("🩶 View General Heart Health Guidance"):
                    st.markdown("""
                    ### ⚖ **Next Steps**
                    - 🩺 Go for medical check-up
                    - 🍎 Maintain healthy diet
                    - 🧘 Practice stress management
                    - 🚶 Keep daily physical activity
                    """)

            else:
                st.success(f"✅ *No Heart Disease Detected* ({positive_count}/{total_models} models indicate heart disease)")
                with st.expander("💪 View Heart Health Maintenance Tips"):
                    st.markdown("""
                    ### 💚 **Keep Your Heart Strong**
                    - 🥦 Eat fiber-rich foods
                    - 🚴 Exercise regularly
                    - 😴 Sleep 7–8 hours daily
                    - 💧 Stay hydrated
                    - 📋 Regular health checkups
                    """)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠ No valid predictions available. Please verify input data.")

# -------------------- TAB 2: BULK PREDICTION (FIXED) -------------------- #
with tab2:
    st.header("📂 Bulk Prediction Upload")
    st.markdown("Upload a CSV file with patient data for bulk heart disease predictions.")
    
    with st.expander("ℹ️ Required CSV Format"):
        st.markdown("""
        Your CSV should have the following columns (in any order):
        - `age`, `sex`, `chest_pain_type`, `resting_bp`, `cholesterol`
        - `fasting_blood_sugar`, `resting_ecg`, `max_heart_rate`
        - `exercise_angina`, `oldpeak`, `st_slope`
        
        **Note:** All values should be numeric (already encoded).
        """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            bulk_data = pd.read_csv(uploaded_file)
            st.write(f"**Total Records:** {len(bulk_data)}")
            st.write("**Preview of Uploaded Data:**")
            st.dataframe(bulk_data.head(10))

            if st.button("🔍 Predict for Uploaded Data"):
                with st.spinner('Processing predictions...'):
                    st.subheader('🩺 Bulk Prediction Results')
                    st.markdown('---')

                    # Get predictions from all models
                    bulk_results = predict_heart_disease(bulk_data)

                    # Create result dataframe
                    result_df = bulk_data.copy()
                    
                    # Convert predictions to DataFrame for easier processing
                    predictions_array = np.array([pred for pred in bulk_results if pred is not None])
                    
                    # Add individual model predictions
                    for idx, algo in enumerate(algonames):
                        if bulk_results[idx] is not None:
                            result_df[f'{algo}_Prediction'] = bulk_results[idx]
                    
                    # Calculate final prediction using majority voting
                    if len(predictions_array) > 0:
                        # Sum across models (axis=0) and apply majority rule
                        majority_votes = np.sum(predictions_array, axis=0)
                        final_predictions = (majority_votes > len(predictions_array) / 2).astype(int)
                        
                        result_df['Final_Prediction'] = final_predictions
                        result_df['Final_Prediction'] = result_df['Final_Prediction'].map({
                            0: 'No Heart Disease', 
                            1: 'Heart Disease'
                        })
                        result_df['Models_Agreement'] = [f"{int(vote)}/{len(predictions_array)}" 
                                                         for vote in majority_votes]

                    st.success(f"✅ Successfully processed {len(bulk_data)} records!")
                    st.dataframe(result_df)

                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        disease_count = sum(final_predictions)
                        st.metric("Heart Disease Cases", disease_count)
                    with col2:
                        no_disease_count = len(final_predictions) - disease_count
                        st.metric("No Heart Disease", no_disease_count)
                    with col3:
                        disease_rate = (disease_count / len(final_predictions)) * 100
                        st.metric("Disease Rate", f"{disease_rate:.1f}%")

                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name='bulk_heart_disease_predictions.csv',
                        mime='text/csv',
                    )
        except Exception as e:
            st.error(f"❌ Error processing the uploaded file: {e}")
            st.info("Please ensure your CSV has the correct column names and data format.")

# -------------------- TAB 3: MODEL INFORMATION (ENHANCED) -------------------- #
with tab3:
    st.header("📊 Model Performance & Information")
    
    # Model accuracies
    data = {
        'Decision Trees': 83.97, 
        'Logistic Regression': 80.86, 
        'Random Forest': 90.23, 
        'Support Vector Machine': 84.22
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Model Accuracy Comparison")
        import plotly.express as px
        import plotly.graph_objects as go
        
        Models = list(data.keys())
        Accuracies = list(data.values())
        df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Models', 'Accuracies'])
        
        # Create bar chart with color gradient
        fig = px.bar(df, y='Accuracies', x='Models', 
                     color='Accuracies',
                     color_continuous_scale='RdYlGn',
                     title='Model Accuracy (%)')
        fig.update_layout(height=400, showlegend=False)
        fig.update_traces(text=df['Accuracies'], textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Best Model")
        best_model = max(data, key=data.get)
        best_accuracy = data[best_model]
        st.metric("Model", best_model)
        st.metric("Accuracy", f"{best_accuracy}%")
        st.info("🏆 Highest performing model for heart disease prediction")
    
    st.markdown("---")
    
    # Model descriptions
    st.subheader("📚 Model Descriptions")
    
    model_info = {
        "Decision Trees": {
            "description": "A tree-like model that makes decisions based on asking a series of questions about the features.",
            "pros": "Easy to interpret, handles non-linear relationships",
            "cons": "Prone to overfitting, can be unstable"
        },
        "Logistic Regression": {
            "description": "A statistical model that uses a logistic function to model binary outcomes.",
            "pros": "Fast, interpretable, works well with linear relationships",
            "cons": "Assumes linear relationship between features"
        },
        "Random Forest": {
            "description": "An ensemble of decision trees that combines multiple trees for better accuracy.",
            "pros": "Reduces overfitting, handles non-linear data well",
            "cons": "Less interpretable, computationally intensive"
        },
        "Support Vector Machine": {
            "description": "Finds the optimal boundary between classes by maximizing the margin.",
            "pros": "Effective in high dimensions, memory efficient",
            "cons": "Sensitive to feature scaling, slower on large datasets"
        }
    }
    
    for model_name, info in model_info.items():
        with st.expander(f"🔍 {model_name} - {data[model_name]}% Accuracy"):
            st.write(f"**Description:** {info['description']}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**✅ Pros:**")
                st.write(info['pros'])
            with col2:
                st.write("**⚠️ Cons:**")
                st.write(info['cons'])
    
    st.markdown("---")
    
    # Feature importance section
    st.subheader("📈 Key Features for Prediction")
    st.markdown("""
    The models analyze the following patient features to make predictions:
    
    1. **Age** - Patient's age in years
    2. **Sex** - Biological sex (Male/Female)
    3. **Chest Pain Type** - Type of chest pain experienced
    4. **Resting Blood Pressure** - Blood pressure at rest (mm Hg)
    5. **Cholesterol** - Serum cholesterol level (mg/dl)
    6. **Fasting Blood Sugar** - Blood sugar level after fasting
    7. **Resting ECG** - Electrocardiogram results at rest
    8. **Max Heart Rate** - Maximum heart rate achieved during exercise
    9. **Exercise Angina** - Chest pain induced by exercise
    10. **Oldpeak** - ST depression induced by exercise
    11. **ST Slope** - Slope of peak exercise ST segment
    """)
    
    st.markdown("---")
    
    # Ensemble prediction explanation
    st.subheader("🤝 Ensemble Prediction Method")
    st.markdown("""
    This application uses an **ensemble approach** combining all four models:
    
    - Each model makes an independent prediction
    - **Majority voting** determines the final result
    - If more than half of the models predict heart disease, the final prediction is positive
    - This approach increases reliability and reduces false predictions
    
    **Example:** If 3 out of 4 models predict heart disease → Final prediction is **Heart Disease**
    """)
    