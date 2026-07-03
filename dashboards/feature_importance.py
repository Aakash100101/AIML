import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_feature_importance(rf_model):
    st.subheader("📈 Feature Importance Analysis")
    st.markdown("This panel shows which clinical measurements contribute most to the **Random Forest** model's predictions. The importances are extracted directly from the trained ensemble of decision trees.")
    
    if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
        # Technical features list in correct training order
        tech_features = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
            'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
            'exercise_angina', 'oldpeak', 'st_slope'
        ]
        
        # User-friendly mapping
        friendly_names = {
            'age': 'Age',
            'sex': 'Gender',
            'chest_pain_type': 'Chest Pain Type',
            'resting_bp': 'Resting Blood Pressure',
            'cholesterol': 'Cholesterol',
            'fasting_blood_sugar': 'Fasting Blood Sugar',
            'resting_ecg': 'Resting ECG',
            'max_heart_rate': 'Maximum Heart Rate',
            'exercise_angina': 'Exercise-Induced Angina',
            'oldpeak': 'ST Depression',
            'st_slope': 'ST Segment Slope'
        }
        
        importances = rf_model.feature_importances_
        
        # Create Feature Importance DataFrame
        df_fi = pd.DataFrame({
            'Technical Feature': tech_features,
            'Feature Name': [friendly_names[f] for f in tech_features],
            'Importance Score': importances,
            'Percentage Contribution': importances * 100
        })
        
        # Sort by importance descending
        df_fi = df_fi.sort_values(by='Importance Score', ascending=False).reset_index(drop=True)
        
        # Verify sum to 1.0 (with safe precision)
        sum_importance = float(np.sum(importances))
        
        # A) Top 5 Features Cards
        st.markdown("#### 🔝 Top 5 Most Influential Clinical Features")
        col_fi1, col_fi2, col_fi3, col_fi4, col_fi5 = st.columns(5)
        cols = [col_fi1, col_fi2, col_fi3, col_fi4, col_fi5]
        for idx in range(min(5, len(df_fi))):
            row = df_fi.iloc[idx]
            cols[idx].metric(
                label=f"#{idx+1}: {row['Feature Name']}",
                value=f"{row['Percentage Contribution']:.1f}%",
                help=f"Importance score: {row['Importance Score']:.4f}"
            )
            
        st.markdown("---")
        
        col_vis1, col_vis2 = st.columns(2)
        
        with col_vis1:
            # A) Interactive Plotly Horizontal Bar Chart
            st.markdown("#### Feature Importance Ranking")
            # Highlight top features (top 3 in pink/crimson, others in indigo)
            colors = ['#f5576c' if i < 3 else '#667eea' for i in range(len(df_fi))]
            
            fig_fi = px.bar(
                df_fi,
                x='Percentage Contribution',
                y='Feature Name',
                orientation='h',
                title='Global Feature Importance Ranking (%)',
                labels={'Percentage Contribution': 'Contribution (%)', 'Feature Name': 'Clinical Feature'},
                text='Percentage Contribution'
            )
            fig_fi.update_traces(
                marker_color=colors,
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            fig_fi.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                margin=dict(l=20, r=40, t=40, b=20),
                height=450
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            
        with col_vis2:
            # C) Cumulative Importance Chart
            st.markdown("#### Cumulative Feature Importance")
            df_fi['Cumulative Contribution'] = df_fi['Percentage Contribution'].cumsum()
            
            fig_cum = px.line(
                df_fi,
                x=df_fi.index + 1,
                y='Cumulative Contribution',
                title='Cumulative Contribution of Features',
                labels={'x': 'Number of Features', 'Cumulative Contribution': 'Cumulative Contribution (%)'},
                markers=True
            )
            fig_cum.update_traces(line_color='#764ba2', marker=dict(size=8))
            fig_cum.update_layout(
                yaxis_range=[0, 105],
                xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                margin=dict(l=20, r=20, t=40, b=20),
                height=450
            )
            st.plotly_chart(fig_cum, use_container_width=True)
            
        st.markdown("---")
        
        # D) Feature Importance Table
        st.markdown("#### Detailed Feature Importance Metrics")
        df_table = df_fi[['Feature Name', 'Importance Score', 'Percentage Contribution']].copy()
        df_table['Importance Score'] = df_table['Importance Score'].map(lambda x: f"{x:.4f}")
        df_table['Percentage Contribution'] = df_table['Percentage Contribution'].map(lambda x: f"{x:.2f}%")
        
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        # 6. Medical Interpretation Panel & 7. Explainability Summary Card
        col_exp1, col_exp2 = st.columns([1, 1])
        
        with col_exp1:
            st.markdown("#### 🩺 Clinical Interpretation of Key Indicators")
            with st.expander("ST Segment Slope", expanded=True):
                st.write("**ST Segment Slope:** Most influential indicator of heart disease risk. An abnormal slope pattern (such as Flat or Downsloping) indicates abnormal heart recovery dynamics under exertion.")
            with st.expander("Maximum Heart Rate", expanded=True):
                st.write("**Maximum Heart Rate:** Lower maximum heart rate achieved during exercise is historically associated with higher cardiovascular risk due to impaired cardiac output capacity.")
            with st.expander("Chest Pain Type", expanded=True):
                st.write("**Chest Pain Type:** A strong predictor of underlying abnormalities. Asymptomatic chest pain is highly correlated with ischemic heart events.")
            with st.expander("ST Depression", expanded=True):
                st.write("**ST Depression (Oldpeak):** Measures exercise-induced cardiac stress. Larger depression values indicate greater local muscle strain and oxygen deficiency.")
            with st.expander("Exercise-Induced Angina", expanded=True):
                st.write("**Exercise-Induced Angina:** A direct indicator of significant heart strain and insufficient blood supply to heart muscle during physical activity.")
                
        with col_exp2:
            st.markdown("#### 🧠 Explainability Summary")
            st.info(
                f"**Explainability Summary:** The Random Forest model primarily relies on "
                f"**{df_fi.iloc[0]['Feature Name']}**, **{df_fi.iloc[1]['Feature Name']}**, "
                f"**{df_fi.iloc[2]['Feature Name']}**, **{df_fi.iloc[3]['Feature Name']}**, and "
                f"**{df_fi.iloc[4]['Feature Name']}** (representing the top 5 features) to classify a patient's risk of heart disease. "
                f"Together, these top 5 features account for **{df_fi.iloc[:5]['Percentage Contribution'].sum():.1f}%** of the model's total predictive influence. "
                f"This indicates that cardiac stress indicators and performance under physical activity carry the largest diagnostic weight in the model's logic."
            )
            
            # Check sum of feature importances
            st.caption(f"🔧 Model Check: Total feature importances sum = {sum_importance:.4f} (expected: 1.0000)")
    else:
        st.error("❌ The Random Forest model is not available or does not contain feature importances.")
