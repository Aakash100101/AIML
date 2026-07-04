import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_model_dashboard(computed_metrics, algonames):
    st.subheader("🤖 Model Performance Leaderboard")
    st.markdown("This dashboard evaluates each classifier dynamically on the clinical cohort dataset ([heart.csv](file:///c:/Users/Aakash/Desktop/heart/data/heart.csv)). Metrics are computed using predictions compared to the ground-truth target (`HeartDisease`).")
    
    if computed_metrics:
        # Highlight best model
        best_model_name = max(
            computed_metrics.keys(),
            key=lambda k: computed_metrics[k]['accuracy'] if computed_metrics[k] is not None else 0
        )
        best_metrics = computed_metrics[best_model_name]
        best_accuracy_val = best_metrics['accuracy'] * 100
        
        # Calculate stats for metric cards
        valid_metrics = [v for v in computed_metrics.values() if v is not None]
        total_models = len(valid_metrics)
        avg_accuracy_val = np.mean([v['accuracy'] for v in valid_metrics]) * 100
        
        # Top Metric Cards
        st.markdown("#### Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏆 Best Model", best_model_name)
        col2.metric("🎯 Best Accuracy", f"{best_accuracy_val:.2f}%")
        col3.metric("📊 Avg Accuracy", f"{avg_accuracy_val:.2f}%")
        col4.metric("🤖 Total Models", total_models)
        
        st.success(f"🏆 **Best Performing Classifier:** **{best_model_name}** with **{best_accuracy_val:.2f}% Accuracy** and **{best_metrics['f1_score']*100:.2f}% F1 Score**.")
        
        # Build Leaderboard Table
        st.markdown("#### Performance Comparison Leaderboard")
        leaderboard_data = []
        for name in algonames:
            m = computed_metrics.get(name)
            if m is not None:
                leaderboard_data.append({
                    "Classifier": name,
                    "Accuracy (%)": round(m['accuracy'] * 100, 2),
                    "Precision (%)": round(m['precision'] * 100, 2),
                    "Recall (%)": round(m['recall'] * 100, 2),
                    "F1 Score (%)": round(m['f1_score'] * 100, 2),
                    "ROC-AUC (%)": round(m['roc_auc'] * 100, 2)
                })
        df_leaderboard = pd.DataFrame(leaderboard_data)
        # Sort by accuracy
        df_leaderboard = df_leaderboard.sort_values(by="Accuracy (%)", ascending=False)
        
        # Show table with style highlight
        st.dataframe(
            df_leaderboard.style.highlight_max(subset=["Accuracy (%)", "F1 Score (%)", "ROC-AUC (%)"], color="rgba(102, 126, 234, 0.2)"),
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        st.markdown("#### Performance Visualizations")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Plotly Grouped Bar Chart
            chart_data = []
            for name in algonames:
                m = computed_metrics.get(name)
                if m is not None:
                    for k, v in m.items():
                        chart_data.append({
                            "Classifier": name,
                            "Metric": k.replace("_", " ").title(),
                            "Value (%)": round(v * 100, 2)
                        })
            df_chart = pd.DataFrame(chart_data)
            
            fig_bar = px.bar(
                df_chart,
                x="Metric",
                y="Value (%)",
                color="Classifier",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Classifier Metric Comparison Matrix"
            )
            fig_bar.update_layout(
                yaxis_title="Value (%)",
                xaxis_title="Evaluation Metric",
                legend_title="Classifier",
                margin=dict(l=20, r=20, t=40, b=20),
                height=450
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_chart2:
            # Plotly Radar Chart
            fig_radar = go.Figure()
            categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
            
            for name in algonames:
                m = computed_metrics.get(name)
                if m is not None:
                    values = [
                        m['accuracy'] * 100,
                        m['precision'] * 100,
                        m['recall'] * 100,
                        m['f1_score'] * 100,
                        m['roc_auc'] * 100
                    ]
                    # Close the loop
                    values.append(values[0])
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories + [categories[0]],
                        fill='toself',
                        name=name
                    ))
                    
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title="Classifier Performance Radar Map",
                height=450,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
    else:
        st.warning("Unable to display model performance. Please verify dataset and model files.")
