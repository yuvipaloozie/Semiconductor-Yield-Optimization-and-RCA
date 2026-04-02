import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.preprocessing import clean_and_prepare_data

st.set_page_config(page_title="Fab Yield Dashboard", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_production_assets():
    model = joblib.load('models/secom_v1_model.pkl')
    with open('models/threshold.txt', 'r') as f:
        threshold = float(f.read())
    top_features = joblib.load('models/top_features.pkl')
    pdp_limits = joblib.load('models/pdp_limits.pkl')
    return model, threshold, top_features, pdp_limits

@st.cache_data
def load_and_prep_data(filepath):
    df_raw = pd.read_csv(filepath)
    X_clean, y = clean_and_prepare_data(df_raw, is_training=False)
    return X_clean, y


def plot_historian_trend(history_df, sensor_name, pdp_data, prob_threshold):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    limit_val = pdp_data['limit']
    direction = pdp_data['direction']
    
    chart_min = history_df[sensor_name].min() * 0.95
    chart_max = history_df[sensor_name].max() * 1.05
    if chart_min == chart_max: chart_min -= 1; chart_max += 1


    safe_color = "rgba(189, 195, 199, 0.2)"   # Muted Grey for normal operation
    danger_color = "rgba(231, 76, 60, 0.1)"    # Muted Red for danger zone
    
    if direction == "Upper":
        fig.add_hrect(y0=chart_min, y1=limit_val, fillcolor=safe_color, layer="below", line_width=0, yref="y")
        fig.add_hrect(y0=limit_val, y1=chart_max, fillcolor=danger_color, layer="below", line_width=0, yref="y")
    else:
        fig.add_hrect(y0=limit_val, y1=chart_max, fillcolor=safe_color, layer="below", line_width=0, yref="y")
        fig.add_hrect(y0=chart_min, y1=limit_val, fillcolor=danger_color, layer="below", line_width=0, yref="y")

   
    fig.add_hline(y=limit_val, line_color="#7F8C8D", line_width=2, line_dash="dash", annotation_text="Operating Limit", secondary_y=False)

    # Sensor Trace (Slate Navy)
    fig.add_trace(
        go.Scatter(x=history_df['Hour'], y=history_df[sensor_name], 
                   mode='lines+markers', line=dict(color='#2C3E50', width=2.5), 
                   name=f'Sensor {sensor_name} (PV)'), secondary_y=False
    )
    
    # ML Probability Trace (Process Blue)
    fig.add_trace(
        go.Scatter(x=history_df['Hour'], y=history_df['Probability'], 
                   mode='lines', line=dict(color='#3498DB', width=2.5), 
                   name='Predicted Risk'), secondary_y=True
    )
    
    # Financial Alarm Line (Red)
    fig.add_hline(y=prob_threshold, line_dash="solid", line_color="#C0392B", 
                  annotation_text="Alarm Threshold", annotation_position="top left", secondary_y=True)
    
    fig.update_layout(
        title=dict(text=f"Parameter {sensor_name}", font=dict(color='#2C3E50', size=16)),
        height=280, margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Time (Hours)", hovermode="x unified",
        template="plotly_white", # Clean white/grey theme
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#2C3E50'))
    )
    
    fig.update_yaxes(title_text="Process Value", secondary_y=False, range=[chart_min, chart_max], gridcolor='#E5E8E8')
    fig.update_yaxes(title_text="Risk Probability", secondary_y=True, range=[0, max(0.2, history_df['Probability'].max() + 0.05)], tickformat='.1%', showgrid=False)
    
    return fig


model, threshold, TOP_SENSORS, pdp_limits = load_production_assets()
X_live, y_live = load_and_prep_data("data/uci-secom.csv")

TOP_3 = TOP_SENSORS[:3]

st.title("Plant Operations Dashboard")
st.markdown("Live SCADA View")

st.sidebar.header("System Controls")
sim_speed = st.sidebar.slider("Simulation Feed Rate (Sec)", 0.1, 2.0, 0.5)
start_button = st.sidebar.button("INITIATE PROCESS MONITOR", type="primary", use_container_width=True)

st.sidebar.markdown("---")

with st.sidebar.expander("Exploratory Model Analysis"):
    st.markdown("Review the static SHAP and PDP diagnostics generated during model training.")
    try:
        st.image("reports/shap_summary.png", caption="Global Feature Importance")
        st.image("reports/partial_dependence.png", caption="Partial Dependence Limits")
    except FileNotFoundError:
        st.warning("Diagnostic reports not found. Run main.py to generate.")


met_col1, met_col2, met_col3 = st.columns(3)
met_card1 = met_col1.empty()
met_card2 = met_col2.empty()
met_card3 = met_col3.empty()

st.markdown("---")

chart_placeholders = {s: st.empty() for s in TOP_3}

st.markdown("### Sensor Log")
log_placeholder = st.empty()

if start_button:
    live_feed = X_live.sample(50).copy()
    live_feed['Actual_Label'] = y_live.loc[live_feed.index].values
    live_feed = live_feed.reset_index(drop=True)
    
    history_df = pd.DataFrame(columns=['Hour', 'Probability'] + TOP_3)
    log_records = []
    
    correct_predictions = 0
    scrap_prevented = 0
    
    for i, row in live_feed.iterrows():
        input_df = pd.DataFrame([row[X_live.columns]])
        prob = model.predict_proba(input_df)[0][1]
        
        pred_label = 1 if prob >= threshold else 0
        actual_label = int(row['Actual_Label'])
        
        is_correct = (pred_label == actual_label)
        if is_correct: correct_predictions += 1
        if pred_label == 1 and actual_label == 1: scrap_prevented += 1
        
        hour_label = f"Hour {i+1}"
        ai_decision = "[ALARM] Scrap Expected" if pred_label == 1 else "[NORMAL] Pass"
        ground_truth = "Fail" if actual_label == 1 else "Pass"
        validation = "Correct" if is_correct else "Miss"
            
        new_row = {'Hour': hour_label, 'Probability': prob}
        for s in TOP_3: new_row[s] = row[s]
        history_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Log Table data
        log_entry = {"Time": hour_label}
        for s in TOP_3: log_entry[f"Sensor {s}"] = round(row[s], 3)
        log_entry.update({
            "Risk Prob": f"{prob:.1%}",
            "System Output": ai_decision,
            "Actual Result": ground_truth,
            "Validation": validation
        })
        log_records.insert(0, log_entry)
      
    
        current_acc = correct_predictions / (i + 1)
        met_card1.metric("Production Uptime", f"{i + 1} Hours")
        met_card2.metric("Inference Accuracy", f"{current_acc:.1%}")
        met_card3.metric("Scrap Events Prevented", scrap_prevented)
        
        plot_df = history_df.tail(15) 
        for s in TOP_3:
            fig_hist = plot_historian_trend(plot_df, s, pdp_limits[s], threshold)
            chart_placeholders[s].plotly_chart(fig_hist, use_container_width=True, key=f"hist_{s}_step_{i}")

        log_placeholder.dataframe(pd.DataFrame(log_records), use_container_width=True, hide_index=True)

        time.sleep(sim_speed)
