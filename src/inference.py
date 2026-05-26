import joblib
import pandas as pd
import time
from src.preprocessing import clean_and_prepare_data

def run_live_simulation(data_path, num_batches=10):
    print("\n Intializing Inference:")
    model = joblib.load('models/secom_v1_model.pkl')
    
    with open('models/threshold.txt', 'r') as f:
        threshold = float(f.read())
    
    df_raw = pd.read_csv(data_path)
    X_clean, _ = clean_and_prepare_data(df_raw, is_training=False)
    raw_feed = X_clean.sample(num_batches)
    
    for i, (idx, row) in enumerate(raw_feed.iterrows()):

        input_df = pd.DataFrame([row]) 
        probability = model.predict_proba(input_df)[0][1]
        status = "FAIL" if probability >= threshold else "PASS"
        
        print(f"[{i+1}/{num_batches}] Batch ID: {idx} | Probability: {probability:.1%} | Status: {status}")
        time.sleep(1.5)
