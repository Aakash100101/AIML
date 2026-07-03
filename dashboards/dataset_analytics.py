import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.preprocessing import preprocess_df

def render_dataset_analytics(df_heart):
    st.subheader("🗃 Dataset Cohort Analytics")
    st.markdown("This dashboard displays exploratory data analysis of the clinical cohort represented by the [heart.csv](file:///c:/Users/Aakash/Desktop/heart/data/heart.csv) dataset (918 patients). Use these charts to understand baseline distributions and cohort relationships.")
    
    if df_heart is not None:
        # 1. Calculations for KPI Cards
        total_patients = len(df_heart)
        disease_cases = int(df_heart['HeartDisease'].sum())
        healthy_cases = total_patients - disease_cases
        disease_rate = (disease_cases / total_patients) * 100
        avg_age = float(df_heart['Age'].mean())
        avg_chol = float(df_heart['Cholesterol'].mean())
        
        # Create KPI Card Grid
        st.markdown("#### Baseline Cohort Metrics")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
        col_kpi1.metric("👥 Total Patients", f"{total_patients}")
        col_kpi2.metric("❤️ Disease Cases", f"{disease_cases}", f"{disease_rate:.1f}% rate", delta_color="inverse")
        col_kpi3.metric("✅ Healthy Cases", f"{healthy_cases}", f"{(100 - disease_rate):.1f}% rate")
        col_kpi4.metric("👶 Avg Age", f"{avg_age:.1f} yrs")
        col_kpi5.metric("🩸 Avg Cholesterol", f"{avg_chol:.1f} mg/dl")
        
        st.markdown("---")
        
        # Visualizations Layout
        df_plot = df_heart.copy()
        df_plot['HeartDisease_Label'] = df_plot['HeartDisease'].map({0: 'Healthy', 1: 'Heart Disease'})
        
        col_row1_1, col_row1_2 = st.columns(2)
        
        with col_row1_1:
            # A) Disease Distribution Pie Chart
            fig_pie = px.pie(
                df_plot, 
                names='HeartDisease_Label', 
                title='Heart Disease Cohort Prevalence (%)',
                color='HeartDisease_Label',
                color_discrete_map={'Healthy': '#51cf66', 'Heart Disease': '#ff6b6b'},
                hole=0.4
            )
            fig_pie.update_traces(textinfo='percent+label', textfont_size=14)
            fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_row1_2:
            # B) Age Distribution Histogram
            fig_age = px.histogram(
                df_plot, 
                x='Age', 
                color='HeartDisease_Label',
                barmode='overlay',
                title='Age Distribution by Heart Disease Cohort',
                color_discrete_map={'Healthy': '#51cf66', 'Heart Disease': '#ff6b6b'},
                opacity=0.75
            )
            fig_age.update_layout(
                xaxis_title='Age (years)', 
                yaxis_title='Patient Count',
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
        col_row2_1, col_row2_2 = st.columns(2)
        
        with col_row2_1:
            # C) Gender Distribution Stacked Bar Chart
            df_gender = df_plot.groupby(['Sex', 'HeartDisease_Label']).size().reset_index(name='Count')
            df_gender['Gender'] = df_gender['Sex'].map({'M': 'Male', 'F': 'Female'})
            fig_gender = px.bar(
                df_gender, 
                x='Gender', 
                y='Count', 
                color='HeartDisease_Label',
                title='Gender Distribution by Heart Disease Status',
                color_discrete_map={'Healthy': '#51cf66', 'Heart Disease': '#ff6b6b'},
                labels={'HeartDisease_Label': 'Status'},
                barmode='stack'
            )
            fig_gender.update_layout(
                xaxis_title='Biological Sex', 
                yaxis_title='Patient Count',
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            st.plotly_chart(fig_gender, use_container_width=True)
            
        with col_row2_2:
            # G) Age vs Max Heart Rate Scatter Plot
            fig_scatter = px.scatter(
                df_plot, 
                x='Age', 
                y='MaxHR', 
                color='HeartDisease_Label',
                title='Age vs Maximum Heart Rate (MaxHR)',
                color_discrete_map={'Healthy': '#51cf66', 'Heart Disease': '#ff6b6b'},
                labels={'MaxHR': 'Maximum Heart Rate (bpm)', 'Age': 'Age (years)', 'HeartDisease_Label': 'Status'}
            )
            fig_scatter.update_layout(
                margin=dict(l=20, r=20, t=40, b=20), 
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.markdown("---")
        
        col_row3_1, col_row3_2 = st.columns(2)
        
        with col_row3_1:
            # D) Cholesterol Distribution Box Plot
            fig_chol = px.box(
                df_plot, 
                x='HeartDisease_Label', 
                y='Cholesterol', 
                color='HeartDisease_Label',
                title='Serum Cholesterol Levels by Cohort',
                color_discrete_map={'Healthy': '#51cf66', 'Heart Disease': '#ff6b6b'},
                labels={'HeartDisease_Label': 'Cohort', 'Cholesterol': 'Serum Cholesterol (mg/dl)'}
            )
            fig_chol.update_layout(
                margin=dict(l=20, r=20, t=40, b=20), 
                height=400
            )
            st.plotly_chart(fig_chol, use_container_width=True)
            
        with col_row3_2:
            # E) Resting Blood Pressure Distribution Box Plot
            fig_bp = px.box(
                df_plot, 
                x='HeartDisease_Label', 
                y='RestingBP', 
                color='HeartDisease_Label',
                title='Resting Blood Pressure Levels by Cohort',
                color_discrete_map={'Healthy': '#51cf66', 'Heart Disease': '#ff6b6b'},
                labels={'HeartDisease_Label': 'Cohort', 'RestingBP': 'Resting Blood Pressure (mm Hg)'}
            )
            fig_bp.update_layout(
                margin=dict(l=20, r=20, t=40, b=20), 
                height=400
            )
            st.plotly_chart(fig_bp, use_container_width=True)
            
        st.markdown("---")
        
        # F) Correlation Heatmap
        st.markdown("#### Clinical Correlation Heatmap")
        st.markdown("This heatmap shows the Pearson correlation coefficients between encoded clinical attributes. Values close to +1.0 or -1.0 signify strong directional relationships.")
        
        df_numeric = preprocess_df(df_heart)
        df_numeric['HeartDisease'] = df_heart['HeartDisease']
        
        friendly_labels = {
            'age': 'Age',
            'sex': 'Gender',
            'chest_pain_type': 'Chest Pain Type',
            'resting_bp': 'Resting Blood Pressure',
            'cholesterol': 'Cholesterol',
            'fasting_blood_sugar': 'Fasting Blood Sugar',
            'resting_ecg': 'Resting ECG',
            'max_heart_rate': 'Max Heart Rate',
            'exercise_angina': 'Exercise Angina',
            'oldpeak': 'ST Depression',
            'st_slope': 'ST Slope',
            'HeartDisease': 'Heart Disease (Target)'
        }
        df_corr = df_numeric.rename(columns=friendly_labels).corr()
        
        fig_corr = px.imshow(
            df_corr,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Pearson Correlation Coefficients (Full Cohort)',
            aspect='auto'
        )
        fig_corr.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        # Clinical Insights Panel
        st.markdown("#### 🩺 Clinical Insights & Patient Cohort Summary")
        
        # Calculate dynamic figures for insights
        male_sub = df_heart[df_heart['Sex'] == 'M']
        female_sub = df_heart[df_heart['Sex'] == 'F']
        male_disease_rate = (male_sub['HeartDisease'].sum() / len(male_sub)) * 100
        female_disease_rate = (female_sub['HeartDisease'].sum() / len(female_sub)) * 100
        
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.info(
                f"📊 **Disease Prevalence & Baseline**\n\n"
                f"- The dataset represents a cohort of **{total_patients} patients**.\n"
                f"- A total of **{disease_cases} patients ({disease_rate:.1f}%)** are positive for heart disease, "
                f"while **{healthy_cases} patients ({100-disease_rate:.1f}%)** are healthy.\n"
                f"- The average age of the cohort is **{avg_age:.1f} years**, with a standard deviation indicating concentration in the 50–65 age bracket, which represents the highest volume of positive cases."
            )
            st.warning(
                f"🧬 **High-Risk Demographic Grouping**\n\n"
                f"- **Gender Disparity**: Male patients represent a significantly high-risk group, with a heart disease rate of **{male_disease_rate:.1f}%** (out of {len(male_sub)} male patients) "
                f"compared to **{female_disease_rate:.1f}%** for female patients (out of {len(female_sub)} female patients).\n"
                f"- **Symptomatic Indicators**: Patients exhibiting Asymptomatic chest pain (ASY) present a significantly higher incidence rate of underlying coronary disease compared to patients with typical/atypical angina."
            )
            
        with col_ins2:
            st.success(
                f"📉 **Key Clinical Correlations**\n\n"
                f"- **ST Segment Slope**: Exhibits a very strong correlation with heart disease. A flat or downsloping ST segment post-exertion is a highly active indicator of disease state.\n"
                f"- **Maximum Heart Rate (MaxHR)**: Demonstrates a strong negative correlation. Lower heart rate ceilings during physical activity often point to arterial blockages or cardiovascular decay.\n"
                f"- **Exercise-Induced Angina & ST Depression**: Both parameters correlate strongly with the positive class, showing that exercise-induced cardiovascular strain correlates directly with ischemic pathology."
            )
            
    else:
        st.warning("Unable to display cohort metrics. Please verify heart.csv is present in the application folder.")
