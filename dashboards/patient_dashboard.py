import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_patient_dashboard(input_data, models, algonames):
    st.subheader("👤 Patient Risk & Ensemble Explainability")
    st.markdown("This panel runs real-time inference on the patient inputs selected in **Tab 1** and aggregates the predictions of all four classifiers. Use this dashboard to understand consensus confidence and clinical risk indicators.")
    
    # 1. Inference setup
    X = input_data.values.astype(float)
    
    # Compute predictions and probabilities for all models
    model_predictions = {}
    model_probabilities = {}
    model_confidences = {}
    
    for name in algonames:
        model = models.get(name)
        if model is not None:
            try:
                # Predict class (0 or 1)
                pred_class = int(model.predict(X)[0])
                model_predictions[name] = pred_class
                
                # Predict probability of positive class
                prob_pos = float(model.predict_proba(X)[0][1])
                model_probabilities[name] = prob_pos
                
                # Confidence in prediction
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
        st.warning("No models are loaded or active. Please configure the models directory.")
        return

    # Calculate collective ensemble metrics
    pos_votes = sum([1 for n in valid_names if model_predictions[n] == 1])
    neg_votes = len(valid_names) - pos_votes
    total_votes = len(valid_names)
    
    # Majority voting decision
    if pos_votes >= 3:
        ensemble_pred_label = "Heart Disease"
        ensemble_pred_class = 1
        vote_color = "#ff6b6b"
    elif pos_votes == 2:
        ensemble_pred_label = "Inconclusive (Tie)"
        ensemble_pred_class = -1
        vote_color = "#fcc419"
    else:
        ensemble_pred_label = "Healthy"
        ensemble_pred_class = 0
        vote_color = "#51cf66"
        
    agreement_pct = (max(pos_votes, neg_votes) / total_votes) * 100
    
    # Risk Score calculation: Average positive class probability across all models
    valid_probs = [model_probabilities[n] for n in valid_names if model_probabilities[n] is not None]
    risk_score = float(np.mean(valid_probs)) * 100 if valid_probs else 0.0
    
    # Determine risk category
    if risk_score <= 30.0:
        risk_category = "LOW RISK"
        risk_color = "#51cf66"
        risk_desc = "Patient falls into a low-risk cohort. Routine wellness checks and maintaining a balanced, active lifestyle are recommended."
    elif risk_score <= 70.0:
        risk_category = "MEDIUM RISK"
        risk_color = "#fcc419"
        risk_desc = "Patient exhibits intermediate signs of cardiovascular risk. Recommend lifestyle adjustments (dietary modification, regular cardiovascular exercises) and semi-annual metabolic profiling."
    else:
        risk_category = "HIGH RISK"
        risk_color = "#ff6b6b"
        risk_desc = "CRITICAL: Patient is in a high-risk cardiovascular cohort. Immediate consultation with a cardiologist, continuous ECG monitoring, and diagnostic imaging (echo/stress test) are strongly advised."

    # Render Layout
    col_left, col_right = st.columns([1, 1])
    
    # ==================== LEFT COLUMN: RISK SCORE & VOTING ====================
    with col_left:
        # C) Patient Risk Score Gauge & Badge
        st.markdown(f"""
        <div style="background-color:rgba(102, 126, 234, 0.05); padding:20px; border-radius:15px; border-left:5px solid {risk_color}; margin-bottom:20px;">
            <h4 style="margin:0; color:#333;">🎯 Ensemble Patient Risk Score</h4>
            <div style="display:flex; align-items:center; margin-top:10px;">
                <span style="font-size:36px; font-weight:700; color:#2b2b2b;">{risk_score:.1f}%</span>
                <span style="background-color:{risk_color}; color:white; padding:4px 12px; border-radius:20px; font-size:14px; font-weight:600; margin-left:15px;">
                    {risk_category}
                </span>
            </div>
            <p style="margin:10px 0 0 0; font-size:14px; color:#555;">{risk_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Plotly Gauge Chart for Risk Score
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
            height=280,
            margin=dict(l=30, r=30, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # B) Ensemble Voting Dashboard
        st.markdown(f"""
        <div style="background-color:rgba(118, 75, 162, 0.05); padding:20px; border-radius:15px; border-left:5px solid {vote_color}; margin-bottom:20px;">
            <h4 style="margin:0; color:#333;">🗳 Consensus Decision Analysis</h4>
            <div style="display:flex; align-items:center; margin-top:10px;">
                <span style="font-size:24px; font-weight:700; color:#2b2b2b;">{pos_votes}/{total_votes} Positive Votes</span>
                <span style="background-color:{vote_color}; color:white; padding:4px 12px; border-radius:20px; font-size:14px; font-weight:600; margin-left:15px;">
                    Consensus: {ensemble_pred_label}
                </span>
            </div>
            <p style="margin:8px 0 0 0; font-size:13px; color:#666;">
                Majority Agreement level: <b>{agreement_pct:.0f}%</b> ({pos_votes} models predict Heart Disease, {neg_votes} predict Healthy).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Plotly Donut Chart of Votes
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
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
        # Agreement Progress Bar
        st.markdown("**Consensus Agreement Strength**")
        st.progress(agreement_pct / 100.0)
        st.caption(f"Strong agreement implies high model alignment ({agreement_pct:.0f}% consensus).")

    # ==================== RIGHT COLUMN: MODEL COMPARISONS & RECS ====================
    with col_right:
        # A) Patient Ensemble Dashboard Predictions
        st.markdown("#### 🤖 Model Inference & Prediction Cards")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            # Model 1
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
            
            # Model 2
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
            # Model 3
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
            
            # Model 4
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
        
        # D) Model Confidence Dashboard with Progress Bars
        st.markdown("#### 📊 Classifier Confidence Dashboard")
        
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
        
        # Display progress bar chart list
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

        # Plotly Horizontal Bar Chart of Confidences
        fig_conf_bar = px.bar(
            df_conf,
            x="Confidence (%)",
            y="Classifier",
            color="Prediction",
            color_discrete_map={'Heart Disease': '#ff6b6b', 'Healthy': '#51cf66'},
            orientation='h',
            title='Inference Confidence Comparison (%)',
            range_x=[0, 105],
            text='Confidence (%)'
        )
        fig_conf_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_conf_bar.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            margin=dict(l=10, r=40, t=40, b=10),
            height=200,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_conf_bar, use_container_width=True)

    st.markdown("---")
    
    # ==================== E) CLINICAL RECOMMENDATION ENGINE ====================
    st.markdown("### 🩺 Clinical Recommendation Engine")
    st.markdown("Recommendations are dynamically compiled based on the ensemble consensus, risk score categorization, and explicit patient physiological indicators.")
    
    # Extract patient variables for recommendation overrides
    p_age = float(input_data['age'].values[0])
    p_sex_num = int(input_data['sex'].values[0])
    p_bp = float(input_data['resting_bp'].values[0])
    p_chol = float(input_data['cholesterol'].values[0])
    p_slope = int(input_data['st_slope'].values[0])
    p_angina = int(input_data['exercise_angina'].values[0])
    
    recs = []
    warnings = []
    
    # 1. Base Recommendations based on Risk Score
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

    # 2. Risk overrides based on patient physiological inputs
    if p_bp > 140.0:
        warnings.append(f"⚠️ **Hypertensive State Detected ({p_bp:.0f} mm Hg):** Patient BP exceeds the normal range. Reducing sodium intake, optimizing potassium intake, and monitoring arterial pressure are indicated.")
    if p_chol > 240.0:
        warnings.append(f"⚠️ **Hypercholesterolemia Risk ({p_chol:.0f} mg/dl):** Highly elevated serum cholesterol. Diet modifications to limit saturated and trans fats, and discussion of lipid-lowering therapies are recommended.")
    if p_slope == 1 or p_slope == 2:  # Flat or Downsloping
        warnings.append("⚠️ **Ischemic Recovery Indicator:** The patient exhibits flat or downsloping post-exercise ST segments. This abnormal electrical recovery is a hallmark of myocardial oxygen deficit during strain.")
    if p_angina == 1:
        warnings.append("⚠️ **Active Anginal Symptom:** Patient experiences chest pain during physical exertion, pointing directly to a mismatch between cardiac oxygen supply and demand.")
    if p_age > 65.0:
        warnings.append(f"⚠️ **Age-Related Risk Factor ({p_age:.0f} yrs):** Advanced age increases vulnerability to coronary atherosclerosis. Diagnostic thresholds should be interpreted with clinical caution.")

    # Render recommendations
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("#### 📋 Recommended Management Action Plan")
        for r in recs:
            st.markdown(r)
            
    with rec_col2:
        st.markdown("#### ⚠️ Highlighted Clinical Warnings")
        if warnings:
            for w in warnings:
                st.markdown(w)
        else:
            st.success("✅ No extreme clinical physiological threshold overrides were triggered. Continue routine screening.")
            
    st.info("**Disclaimer:** This dashboard is an explainable AI tool designed to assist clinical research and education. It does not replace professional medical diagnosis, treatment, or judgment.")

    # Model comparison table
    st.markdown("---")
    st.markdown("#### 📋 Detailed Patient Prediction Matrix")
    
    matrix_data = []
    for name in algonames:
        pred_label = "Heart Disease" if model_predictions[name] == 1 else "Healthy"
        prob_val = model_probabilities[name]
        conf_val = model_confidences[name] * 100
        matrix_data.append({
            "Classifier Model": name,
            "Prediction": pred_label,
            "Positive Class Prob (Heart Disease)": f"{prob_val:.4f}",
            "Consensus Vote Weight": "1.0",
            "Inference Confidence": f"{conf_val:.2f}%"
        })
    df_matrix = pd.DataFrame(matrix_data)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)
