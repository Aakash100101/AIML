import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Import PDF generator helper
from utils.pdf_generator import generate_pdf_report

def render_patient_dashboard(input_data, models, algonames, df_heart):
    # 1. Setup Inference inputs
    X = input_data.values.astype(float)
    
    # 2. Run model predictions, probabilities and confidences
    model_predictions = {}
    model_probabilities = {}
    model_confidences = {}
    
    for name in algonames:
        model = models.get(name)
        if model is not None:
            try:
                pred_class = int(model.predict(X)[0])
                model_predictions[name] = pred_class
                
                prob_pos = float(model.predict_proba(X)[0][1])
                model_probabilities[name] = prob_pos
                
                conf = prob_pos if pred_class == 1 else (1.0 - prob_pos)
                model_confidences[name] = conf
            except Exception as e:
                st.error(f"Error predicting with {name}: {e}")
                model_predictions[name] = None
                model_probabilities[name] = None
                model_confidences[name] = None
        else:
            model_predictions[name] = None
            model_probabilities[name] = None
            model_confidences[name] = None

    valid_names = [n for n in algonames if model_predictions[n] is not None]
    
    if not valid_names:
        st.warning("No models are loaded or active. Please check the models directory.")
        return

    # Ensemble calculations
    pos_votes = sum([1 for n in valid_names if model_predictions[n] == 1])
    neg_votes = len(valid_names) - pos_votes
    total_votes = len(valid_names)
    
    if pos_votes >= 3:
        ensemble_label = "Heart Disease"
        ensemble_color = "#ff6b6b"
    elif pos_votes == 2:
        ensemble_label = "Inconclusive (Tie)"
        ensemble_color = "#fcc419"
    else:
        ensemble_label = "Healthy"
        ensemble_color = "#51cf66"
        
    agreement_pct = (max(pos_votes, neg_votes) / total_votes) * 100
    valid_probs = [model_probabilities[n] for n in valid_names if model_probabilities[n] is not None]
    risk_score = float(np.mean(valid_probs)) * 100 if valid_probs else 0.0

    if risk_score <= 30.0:
        risk_category = "LOW RISK"
        risk_color = "#51cf66"
        risk_desc = "Patient is in the low-risk range. Continuous lifestyle maintenance is recommended."
    elif risk_score <= 70.0:
        risk_category = "MEDIUM RISK"
        risk_color = "#fcc419"
        risk_desc = "Patient is in the intermediate risk range. Lifestyle improvements and monitoring are recommended."
    else:
        risk_category = "HIGH RISK"
        risk_color = "#ff6b6b"
        risk_desc = "Patient is in the high-risk range. Cardiological screening and diagnostics are strongly advised."

    # 3. Patient Clinical variables extraction & warning thresholds
    p_age = float(input_data['age'].values[0])
    p_sex_num = int(input_data['sex'].values[0])
    p_bp = float(input_data['resting_bp'].values[0])
    p_chol = float(input_data['cholesterol'].values[0])
    p_slope = int(input_data['st_slope'].values[0])
    p_angina = int(input_data['exercise_angina'].values[0])
    p_oldpeak = float(input_data['oldpeak'].values[0])
    p_cp = int(input_data['chest_pain_type'].values[0])
    
    recs = []
    warnings = []
    
    # Base Recommendations based on Risk Score
    if risk_score <= 30.0:
        recs.append("🏃 **Cardiovascular Wellness Routine:** Continue engaging in standard aerobic activity (minimum 150 minutes per week of moderate-intensity activity like brisk walking).")
        recs.append("🥗 **Balanced Diet Maintenance:** Maintain low-sodium, nutrient-dense Mediterranean diet principles focusing on whole grains, leafy greens, and lean proteins.")
        recs.append("🗓️ **Routine Screenings:** Conduct standard annual physical exams and baseline lipid profiling.")
    elif risk_score <= 70.0:
        recs.append("🩺 **Clinical Evaluation:** Schedule a follow-up assessment with a primary care practitioner to discuss sub-clinical risk profiles.")
        recs.append("🔄 **Targeted Lifestyle Modifications:** Formulate a structured physical training and dietary regimen to reduce peripheral vascular resistance and manage blood lipids.")
        recs.append("📊 **Home Monitoring:** Consider monitoring resting blood pressure weekly and tracking active heart rates during exercise.")
    else:
        recs.append("🚨 **Urgent Cardiologist Consult:** Arrange a consultation with a cardiologist to review this high-risk prediction cohort result.")
        recs.append("⚡ **Cardiac Stress Testing:** Schedule a clinical graded exercise test (stress test) or echocardiogram to evaluate active myocardial perfusion.")
        recs.append("📋 **Diagnostic Panel:** Plan for a high-sensitivity C-reactive protein (hs-CRP) and ambulatory ECG screening.")
        recs.append("💊 **Pharmacotherapy Review:** Discuss risk-reduction pharmacotherapies (such as statins or anti-hypertensives) with a licensed physician.")

    # Physiological overrides warnings
    if p_bp > 140.0:
        warnings.append(f"⚠️ **Hypertensive State Detected ({p_bp:.0f} mm Hg):** Patient BP exceeds the normal range. Reducing sodium intake, optimizing potassium intake, and monitoring pressure are indicated.")
    if p_chol > 240.0:
        warnings.append(f"⚠️ **Hypercholesterolemia Risk ({p_chol:.0f} mg/dl):** Highly elevated serum cholesterol. Diet modifications to limit saturated and trans fats are recommended.")
    if p_slope == 1 or p_slope == 2:  # Flat or Downsloping
        warnings.append("⚠️ **Ischemic Recovery Indicator:** The patient exhibits flat or downsloping post-exercise ST segments. This abnormal electrical recovery is a hallmark of myocardial oxygen deficit during strain.")
    if p_angina == 1:
        warnings.append("⚠️ **Active Anginal Symptom:** Patient experiences chest pain during physical exertion, pointing directly to a mismatch between cardiac oxygen supply and demand.")
    if p_age > 65.0:
        warnings.append(f"⚠️ **Age-Related Risk Factor ({p_age:.0f} yrs):** Advanced age increases vulnerability to coronary atherosclerosis. Diagnostic thresholds should be interpreted with clinical caution.")

    # 4. Decision Tree Path Rules compilation
    feature_names = [
        'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
        'exercise_angina', 'oldpeak', 'st_slope'
    ]
    friendly_names = {
        'age': 'Age',
        'sex': 'Gender',
        'chest_pain_type': 'Chest Pain Type',
        'resting_bp': 'Resting Blood Pressure',
        'cholesterol': 'Serum Cholesterol',
        'fasting_blood_sugar': 'Fasting Blood Sugar',
        'resting_ecg': 'Resting ECG',
        'max_heart_rate': 'Maximum Heart Rate',
        'exercise_angina': 'Exercise-Induced Angina',
        'oldpeak': 'ST Depression (Oldpeak)',
        'st_slope': 'ST Segment Slope'
    }
    
    rules = []
    dt_model = models.get('Decision Trees')
    if dt_model is not None:
        try:
            node_indicator = dt_model.decision_path(X)
            node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
            tree = dt_model.tree_
            for node_id in node_index:
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    continue
                f_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                f_name = feature_names[f_idx]
                val = X[0, f_idx]
                
                if val <= threshold:
                    sign = "<="
                else:
                    sign = ">"
                    
                med_name = friendly_names.get(f_name, f_name)
                
                if f_name == 'chest_pain_type':
                    desc = "Chest Pain is Symptomatic (TA, ATA, or NAP)" if sign == "<=" else "Chest Pain is Asymptomatic (ASY - High Risk)"
                elif f_name == 'st_slope':
                    desc = "ST Slope is Upsloping (Normal)" if sign == "<=" else "ST Slope is Flat or Downsloping (Abnormal)"
                elif f_name == 'exercise_angina':
                    desc = "No Exercise-Induced Angina" if sign == "<=" else "Exercise-Induced Angina is Present (Abnormal)"
                elif f_name == 'sex':
                    desc = "Gender is Male" if sign == "<=" else "Gender is Female"
                elif f_name == 'fasting_blood_sugar':
                    desc = "Fasting Blood Sugar <= 120 mg/dl" if sign == "<=" else "Fasting Blood Sugar > 120 mg/dl (Elevated)"
                else:
                    desc = f"{med_name} {sign} {threshold:.1f} (Patient: {val:.1f})"
                rules.append(desc)
        except Exception as e:
            st.error(f"Error compiling Decision Tree rules: {e}")

    # 5. Logistic Regression contributions compilation
    df_pos = None
    df_neg = None
    lr_model = models.get('Logistic Regression')
    if lr_model is not None:
        try:
            coefs = lr_model.coef_[0]
            features_raw = [
                'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
                'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
                'exercise_angina', 'oldpeak', 'st_slope'
            ]
            contributions = coefs * X[0]
            df_contrib = pd.DataFrame({
                "Feature": [friendly_names[f] for f in features_raw],
                "Coefficient": coefs,
                "Patient Value": X[0],
                "Contribution": contributions
            })
            df_contrib = df_contrib.sort_values(by="Contribution", ascending=False)
            df_pos = df_contrib[df_contrib["Contribution"] > 0]
            df_neg = df_contrib[df_contrib["Contribution"] < 0]
        except Exception as e:
            st.error(f"Error compiling Logistic Regression contributions: {e}")

    # 6. Patient Raw details dictionary for PDF layout
    patient_raw_dict = {
        'Age': f"{int(p_age)} years",
        'Gender': "Male" if p_sex_num == 0 else "Female",
        'Chest Pain Type': ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"][p_cp],
        'Resting Blood Pressure': f"{int(p_bp)} mm Hg",
        'Serum Cholesterol': f"{int(p_chol)} mg/dl",
        'Fasting Blood Sugar': "> 120 mg/dl" if int(input_data['fasting_blood_sugar'].values[0]) == 1 else "<= 120 mg/dl",
        'Resting ECG': ["Normal", "ST wave abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"][int(input_data['resting_ecg'].values[0])],
        'Maximum Heart Rate': f"{int(input_data['max_heart_rate'].values[0])} bpm",
        'Exercise-Induced Angina': "Yes" if p_angina == 1 else "No",
        'ST Depression (Oldpeak)': f"{p_oldpeak:.1f} mm",
        'ST Segment Slope': ["Upsloping (Up)", "Flat", "Downsloping (Down)"][p_slope]
    }

    # Compile the PDF report dynamically in-memory
    try:
        pdf_bytes = generate_pdf_report(
            patient_raw_dict,
            model_predictions,
            model_probabilities,
            model_confidences,
            ensemble_label,
            pos_votes,
            total_votes,
            risk_score,
            risk_category,
            risk_desc,
            rules,
            df_pos,
            df_neg,
            warnings,
            recs,
            warnings
        )
    except Exception as e:
        st.error(f"Error generating PDF report bytes: {e}")
        pdf_bytes = None

    # LAYOUT GRID
    col_header, col_download = st.columns([3, 1])
    with col_header:
        st.markdown("<h3 style='margin:0; padding:0;'>👤 Patient Inference & Custom Explainable AI (XAI) Panel</h3>", unsafe_allow_html=True)
    with col_download:
        if pdf_bytes is not None:
            st.download_button(
                label="📥 Download Clinical Report",
                data=pdf_bytes,
                file_name=f"Clinical_Heart_Disease_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("PDF compilation failed.")

    # RENDER SUBTABS
    xai_tab1, xai_tab2 = st.tabs([
        "🧬 Patient Risk & Consensus",
        "🔬 Explainable AI (XAI)"
    ])

    # ==================== SUBTAB 1: RISK & CONSENSUS ====================
    with xai_tab1:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown(f"""
            <div style="background-color:rgba(102, 126, 234, 0.05); padding:20px; border-radius:15px; border-left:5px solid {risk_color}; margin-bottom:20px;">
                <h4 style="margin:0; color:#333;">🎯 Patient Risk Score</h4>
                <div style="display:flex; align-items:center; margin-top:10px;">
                    <span style="font-size:36px; font-weight:700; color:#2b2b2b;">{risk_score:.1f}%</span>
                    <span style="background-color:{risk_color}; color:white; padding:4px 12px; border-radius:20px; font-size:14px; font-weight:600; margin-left:15px;">
                        {risk_category}
                    </span>
                </div>
                <p style="margin:10px 0 0 0; font-size:14px; color:#555;">{risk_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                number={'suffix': "%", 'font': {'size': 24}},
                title={'text': "Cardiovascular Cohort Risk Index", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#667eea", 'thickness': 0.25},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(81, 207, 102, 0.2)'},
                        {'range': [30, 70], 'color': 'rgba(252, 196, 25, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(255, 107, 107, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=30, r=30, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            
            st.markdown(f"""
            <div style="background-color:rgba(118, 75, 162, 0.05); padding:20px; border-radius:15px; border-left:5px solid {ensemble_color}; margin-bottom:20px;">
                <h4 style="margin:0; color:#333;">🗳 Consensus Decision Analysis</h4>
                <div style="display:flex; align-items:center; margin-top:10px;">
                    <span style="font-size:24px; font-weight:700; color:#2b2b2b;">{pos_votes}/{total_votes} Positive Votes</span>
                    <span style="background-color:{ensemble_color}; color:white; padding:4px 12px; border-radius:20px; font-size:14px; font-weight:600; margin-left:15px;">
                        Consensus: {ensemble_label}
                    </span>
                </div>
                <p style="margin:8px 0 0 0; font-size:13px; color:#666;">
                    Majority Agreement level: <b>{agreement_pct:.0f}%</b> ({pos_votes} models predict Heart Disease, {neg_votes} predict Healthy).
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Heart Disease Votes', 'Healthy Votes'],
                values=[pos_votes, neg_votes],
                hole=0.5,
                marker_colors=['#ff6b6b', '#51cf66'],
                textinfo='value+percent',
                hoverinfo='label+percent+value'
            )])
            fig_donut.update_layout(
                title_text="Consensus Voting Distribution",
                title_x=0.0,
                title_font_size=16,
                height=230,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_right:
            st.markdown("#### 🤖 Model Inference & Prediction Cards")
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                dt_label = "🔴 Heart Disease" if model_predictions['Decision Trees'] == 1 else "🟢 Healthy"
                dt_prob = model_probabilities['Decision Trees']
                dt_conf = model_confidences['Decision Trees'] * 100
                st.markdown(f"""
                <div style="background-color:white; padding:15px; border-radius:12px; border:1px solid #ddd; border-top:4px solid #4dabf7; margin-bottom:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <h5 style="margin:0 0 5px 0; color:#555;">Decision Trees</h5>
                    <div style="font-size:16px; font-weight:700; color:#333;">{dt_label}</div>
                    <div style="font-size:13px; color:#666; margin-top:4px;">
                        Probability: <b>{dt_prob:.4f}</b><br>
                        Confidence: <b>{dt_conf:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                rf_label = "🔴 Heart Disease" if model_predictions['Random Forest'] == 1 else "🟢 Healthy"
                rf_prob = model_probabilities['Random Forest']
                rf_conf = model_confidences['Random Forest'] * 100
                st.markdown(f"""
                <div style="background-color:white; padding:15px; border-radius:12px; border:1px solid #ddd; border-top:4px solid #748ffc; margin-bottom:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <h5 style="margin:0 0 5px 0; color:#555;">Random Forest</h5>
                    <div style="font-size:16px; font-weight:700; color:#333;">{rf_label}</div>
                    <div style="font-size:13px; color:#666; margin-top:4px;">
                        Probability: <b>{rf_prob:.4f}</b><br>
                        Confidence: <b>{rf_conf:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_m2:
                lr_label = "🔴 Heart Disease" if model_predictions['Logistic Regression'] == 1 else "🟢 Healthy"
                lr_prob = model_probabilities['Logistic Regression']
                lr_conf = model_confidences['Logistic Regression'] * 100
                st.markdown(f"""
                <div style="background-color:white; padding:15px; border-radius:12px; border:1px solid #ddd; border-top:4px solid #37b24d; margin-bottom:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <h5 style="margin:0 0 5px 0; color:#555;">Logistic Regression</h5>
                    <div style="font-size:16px; font-weight:700; color:#333;">{lr_label}</div>
                    <div style="font-size:13px; color:#666; margin-top:4px;">
                        Probability: <b>{lr_prob:.4f}</b><br>
                        Confidence: <b>{lr_conf:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                svc_label = "🔴 Heart Disease" if model_predictions['Support Vector Machine'] == 1 else "🟢 Healthy"
                svc_prob = model_probabilities['Support Vector Machine']
                svc_conf = model_confidences['Support Vector Machine'] * 100
                st.markdown(f"""
                <div style="background-color:white; padding:15px; border-radius:12px; border:1px solid #ddd; border-top:4px solid #f76707; margin-bottom:15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <h5 style="margin:0 0 5px 0; color:#555;">SVM Classifier</h5>
                    <div style="font-size:16px; font-weight:700; color:#333;">{svc_label}</div>
                    <div style="font-size:13px; color:#666; margin-top:4px;">
                        Probability: <b>{svc_prob:.4f}</b><br>
                        Confidence: <b>{svc_conf:.1f}%</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("---")
            
            st.markdown("#### 📊 Classifier Confidence Ranking")
            df_conf_list = []
            for name in algonames:
                c = model_confidences[name]
                if c is not None:
                    df_conf_list.append({
                        "Classifier": name,
                        "Confidence (%)": round(c * 100, 2),
                        "Prediction": "Heart Disease" if model_predictions[name] == 1 else "Healthy"
                    })
            df_conf = pd.DataFrame(df_conf_list).sort_values(by="Confidence (%)", ascending=False)
            
            for idx, row in df_conf.iterrows():
                c_val = row["Confidence (%)"]
                pred_val = row["Prediction"]
                lbl_color = "#ff6b6b" if pred_val == "Heart Disease" else "#51cf66"
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:13px; font-weight:600; color:#555;">
                        <span>🤖 {row['Classifier']}</span>
                        <span>Prediction: <b style="color:{lbl_color};">{pred_val}</b> | Confidence: <b>{c_val:.1f}%</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(c_val / 100.0)

    # ==================== SUBTAB 2: EXPLAINABLE AI (XAI) ====================
    with xai_tab2:
        st.markdown("### 🔍 Model Explanations & Path Tracing")
        
        # -------------------- SECTION A: DECISION TREE PATH EXPLAINER --------------------
        st.markdown("#### 🌳 Decision Tree Traced Path Explainer")
        if dt_model is not None and len(rules) > 0:
            st.write("Below is the exact step-by-step split path navigated by the patient's record through the Decision Tree nodes:")
            st.markdown("""
            <style>
                .timeline-container {
                    border-left: 3px solid #667eea;
                    padding-left: 20px;
                    margin-left: 10px;
                    margin-top: 15px;
                }
                .timeline-node {
                    position: relative;
                    margin-bottom: 12px;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #748ffc;
                }
                .timeline-node::before {
                    content: '';
                    position: absolute;
                    left: -27px;
                    top: 15px;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: #667eea;
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
            for idx, r in enumerate(rules):
                st.markdown(f"""
                <div class="timeline-node">
                    <small style="color:#777; font-weight:600;">Node #{idx+1} Split Criteria</small>
                    <div style="font-size:14px; font-weight:500; color:#333;">{r}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.expander("🔬 Complete Diagnostic Path Logical Rules"):
                st.code("IF " + "\nAND ".join(rules) + f"\nTHEN Prediction = {'Heart Disease' if model_predictions['Decision Trees'] == 1 else 'Healthy'}", language="python")
        else:
            st.warning("Decision Tree rules are not available.")
            
        st.markdown("---")
        
        # -------------------- SECTION B: LOGISTIC REGRESSION CONTRIBUTIONS --------------------
        st.markdown("#### ⚖️ Logistic Regression Feature Contributions")
        if df_pos is not None and df_neg is not None:
            try:
                # Plotly Bi-directional contribution bar chart
                df_contrib = pd.concat([df_pos, df_neg]).sort_values(by="Contribution", ascending=False)
                fig_contrib = px.bar(
                    df_contrib,
                    x="Contribution",
                    y="Feature",
                    orientation="h",
                    title="Directional Influence of Clinical Indicators",
                    color="Contribution",
                    color_continuous_scale="RdYlGn_r",
                    labels={"Contribution": "Influence on Risk Index", "Feature": "Clinical Indicator"}
                )
                fig_contrib.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=350,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
                
                # Render Positive and Negative lists
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.markdown("<span style='color:#ff6b6b; font-weight:700;'>🔴 Risk Increasing Indicators (Positive Contribution)</span>", unsafe_allow_html=True)
                    for _, row in df_pos.iterrows():
                        st.write(f"- **{row['Feature']}**: +{row['Contribution']:.4f} (value: {row['Patient Value']:.1f})")
                        
                with col_c2:
                    st.markdown("<span style='color:#51cf66; font-weight:700;'>🟢 Risk Decreasing Indicators (Negative Contribution)</span>", unsafe_allow_html=True)
                    for _, row in df_neg.iterrows():
                        st.write(f"- **{row['Feature']}**: {row['Contribution']:.4f} (value: {row['Patient Value']:.1f})")
            except Exception as e:
                st.error(f"Error compiling Logistic Regression contributions: {e}")
        else:
            st.warning("Logistic Regression contributions are not available.")
            
        st.markdown("---")
        
        # -------------------- SECTION C: PATIENT VS COHORT ANALYSIS --------------------
        st.markdown("#### 📊 Patient vs Cohort Demographics Comparison")
        if df_heart is not None:
            try:
                df_healthy = df_heart[df_heart['HeartDisease'] == 0]
                df_disease = df_heart[df_heart['HeartDisease'] == 1]
                
                cohort_features = [
                    ('Age', 'age', 'years'),
                    ('RestingBP', 'resting_bp', 'mm Hg'),
                    ('Cholesterol', 'cholesterol', 'mg/dl'),
                    ('MaxHR', 'max_heart_rate', 'bpm'),
                    ('Oldpeak', 'oldpeak', 'ST depression')
                ]
                
                col_co1, col_co2, col_co3, col_co4, col_co5 = st.columns(5)
                cols_grid = [col_co1, col_co2, col_co3, col_co4, col_co5]
                
                comparison_data = []
                for idx, (orig_col, app_col, unit) in enumerate(cohort_features):
                    p_val = float(input_data[app_col].values[0])
                    h_mean = float(df_healthy[orig_col].mean())
                    d_mean = float(df_disease[orig_col].mean())
                    
                    if orig_col == 'MaxHR':
                        if p_val < d_mean:
                            status_text = "Critical (Below Disease Avg)"
                            status_col = "#ff6b6b"
                        elif p_val > h_mean:
                            status_text = "Excellent (Above Healthy Avg)"
                            status_col = "#51cf66"
                        else:
                            status_text = "Borderline/Intermediate"
                            status_col = "#fcc419"
                    else:
                        if p_val > d_mean:
                            status_text = "Elevated (Above Disease Avg)"
                            status_col = "#ff6b6b"
                        elif p_val < h_mean:
                            status_text = "Normal (Below Healthy Avg)"
                            status_col = "#51cf66"
                        else:
                            status_text = "Borderline/Intermediate"
                            status_col = "#fcc419"
                            
                    cols_grid[idx].markdown(f"""
                    <div style="background-color:white; padding:12px; border-radius:10px; border:1px solid #eee; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.01); height:190px;">
                        <h6 style="margin:0 0 5px 0; color:#555; font-size:13px;">{orig_col}</h6>
                        <div style="font-size:20px; font-weight:700; color:#2b2b2b;">{p_val:.1f}</div>
                        <div style="font-size:11px; color:#888; margin-top:2px;">Healthy Avg: <b>{h_mean:.1f}</b><br>Disease Avg: <b>{d_mean:.1f}</b></div>
                        <div style="margin-top:10px; font-size:11px; font-weight:600; color:{status_col};">{status_text}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    comparison_data.append({
                        "Metric": orig_col,
                        "Patient": p_val,
                        "Healthy Average": round(h_mean, 1),
                        "Disease Average": round(d_mean, 1)
                    })
                
                df_comp = pd.DataFrame(comparison_data)
                df_melt = df_comp.melt(id_vars="Metric", value_vars=["Patient", "Healthy Average", "Disease Average"], var_name="Group", value_name="Value")
                
                fig_comp = px.bar(
                    df_melt,
                    x="Metric",
                    y="Value",
                    color="Group",
                    barmode="group",
                    title="Active Patient Values vs. Cohort Benchmarks",
                    color_discrete_sequence=['#667eea', '#51cf66', '#ff6b6b']
                )
                fig_comp.update_layout(
                    height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                st.error(f"Error building cohort comparison: {e}")
        else:
            st.warning("Cohort metrics unavailable.")
            
        st.markdown("---")
        
        # -------------------- SECTION D: CLINICAL RISK FACTOR DETECTION --------------------
        st.markdown("#### 🔬 Automated Clinical Risk Factor Detection")
        
        detected_warnings = []
        if p_bp > 140.0:
            detected_warnings.append({
                "factor": "Hypertension Detected",
                "badge": "🔴 High Pressure",
                "color": "#ff6b6b",
                "explanation": f"Patient resting blood pressure is {p_bp:.0f} mm Hg (threshold > 140). Elevated pressure forces the ventricles to exert higher mechanical force to sustain output."
            })
        if p_chol > 240.0:
            detected_warnings.append({
                "factor": "Hypercholesterolemia Detected",
                "badge": "🔴 Elevated Lipids",
                "color": "#ff6b6b",
                "explanation": f"Patient serum cholesterol level is {p_chol:.0f} mg/dl (threshold > 240). Elevated circulating lipids are associated with atherosclerotic plaque deposition."
            })
        if p_angina == 1:
            detected_warnings.append({
                "factor": "Exercise-Induced Angina",
                "badge": "🔴 Ischemic Pain",
                "color": "#ff6b6b",
                "explanation": "Active symptoms of chest discomfort during physical exertion suggest a mismatch between metabolic oxygen demand and arterial supply."
            })
        if p_slope == 1 or p_slope == 2:
            slope_name = "Flat" if p_slope == 1 else "Downsloping"
            detected_warnings.append({
                "factor": f"Abnormal ST Segment Slope ({slope_name})",
                "badge": "🔴 Abnormal Recovery",
                "color": "#ff6b6b",
                "explanation": "Traversing flat or downsloping ST segments post-exertion indicates incomplete electrical recovery of the myocardial tissue."
            })
        if p_oldpeak > 1.5:
            detected_warnings.append({
                "factor": f"Elevated ST Depression (Oldpeak: {p_oldpeak:.1f})",
                "badge": "🔴 Severe Stress",
                "color": "#ff6b6b",
                "explanation": "An ST depression magnitude greater than 1.5 mm under cardiac stress indicates severe demand ischemia."
            })
        if p_cp == 3:
            detected_warnings.append({
                "factor": "Asymptomatic Chest Pain Pattern",
                "badge": "🔴 Silent Ischemia Risk",
                "color": "#ff6b6b",
                "explanation": "The patient reports Asymptomatic chest pain (ASY). Clinically, this is associated with a high prevalence of silent myocardial ischemia."
            })
            
        if detected_warnings:
            col_w1, col_w2 = st.columns(2)
            for i, w in enumerate(detected_warnings):
                col_target = col_w1 if i % 2 == 0 else col_w2
                col_target.markdown(f"""
                <div style="background-color:rgba(255, 107, 107, 0.05); padding:15px; border-radius:10px; border-left:4px solid {w['color']}; margin-bottom:15px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b style="font-size:14px; color:#2b2b2b;">{w['factor']}</b>
                        <span style="background-color:{w['color']}; color:white; padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600;">{w['badge']}</span>
                    </div>
                    <p style="margin:8px 0 0 0; font-size:12px; color:#555; line-height:1.4;">{w['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No cardiovascular clinical risk thresholds were exceeded.")
            
        st.markdown("---")
        
        # -------------------- SECTION E: AI EXPLANATION SUMMARY --------------------
        st.markdown("#### 🧠 AI Clinical Explanation Synthesis")
        top_drivers = []
        if df_pos is not None and not df_pos.empty:
            pos_contrib = df_pos.head(3)
            for _, row in pos_contrib.iterrows():
                top_drivers.append(row['Feature'])
        else:
            if p_slope in [1, 2]: top_drivers.append("ST Slope (Flat/Down)")
            if p_angina == 1: top_drivers.append("Exercise Angina")
            if p_cp == 3: top_drivers.append("Asymptomatic Pain Type")
            
        drivers_str = "• " + "\n• ".join(top_drivers) if top_drivers else "• General age and clinical metrics"
        
        st.info(f"""
        **AI Decision Rationale:**
        
        The predictive models arrived at this patient's cardiovascular classification primarily driven by the following influential indicators:
        
        {drivers_str}
        
        These parameters represent the strongest active risk factors. The collective consensus of **{pos_votes}/{total_votes} models** indicates an overall patient prediction of **{ensemble_label.upper()}** with a cohort risk rating of **{risk_score:.1f}% ({risk_category})**.
        """)
        
        st.markdown("---")
        
        # -------------------- SECTION F: CLINICAL DECISION SUPPORT SUMMARY --------------------
        st.markdown("#### 🏥 Clinical Decision Support Summary")
        
        if risk_category == "LOW RISK":
            st.success("""
            **📋 Low Risk Guidelines:**
            - **Recommendation**: Continue standard cardio wellness screening.
            - **Lifestyle**: Maintain a healthy balanced diet (low sodium) and active lifestyle (150 minutes of aerobic exercise weekly).
            - **Screening**: Annual routine physical checking and lipid profile tracking.
            """)
        elif risk_category == "MEDIUM RISK":
            st.warning("""
            **📋 Medium Risk Guidelines:**
            - **Recommendation**: Initiate targeted lifestyle modifications and periodic checkups.
            - **Lifestyle**: Moderate dietary changes, and engage in regular, monitored cardiovascular exercises.
            - **Screening**: Semi-annual metabolic profiling and weekly resting blood pressure checks.
            """)
        else:
            st.error("""
            **🚨 Critical Care High Risk Guidelines:**
            - **Recommendation**: Immediate cardiologist referral and diagnostics are indicated.
            - **Diagnostics**: Graded treadmill stress test, echocardiogram, or computed tomography coronary angiography (CTCA).
            - **Intervention**: Discuss lipid-lowering (statin) and antihypertensive pharmacotherapies.
            - **Monitoring**: Continuous ECG tracking and immediate symptom alert response.
            """)
            
    st.info("**Explainable AI Disclosure:** This diagnostic path explainer is an educational research aid. It maps machine learning logic to medical terms but does not replace professional clinical decision making.")
    
    # Detailed Patient prediction matrix
    st.markdown("---")
    st.markdown("#### 📋 Detailed Patient Prediction Matrix")
    matrix_data = []
    for name in algonames:
        p_label = "Heart Disease" if model_predictions[name] == 1 else "Healthy"
        prob_val = model_probabilities[name]
        conf_val = model_confidences[name] * 100
        matrix_data.append({
            "Classifier Model": name,
            "Prediction": p_label,
            "Positive Class Prob (Heart Disease)": f"{prob_val:.4f}",
            "Consensus Vote Weight": "1.0",
            "Inference Confidence": f"{conf_val:.2f}%"
        })
    df_matrix = pd.DataFrame(matrix_data)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)
