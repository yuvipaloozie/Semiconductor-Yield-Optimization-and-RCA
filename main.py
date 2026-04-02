import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path: sys.path.append(str(ROOT_DIR))

from src.preprocessing import clean_and_prepare_data
from src.model_training import train_production_model
from src.evaluation import run_full_evaluation
from src.inference import run_live_simulation

def main():
    DATA_PATH = "data/uci-secom.csv"
    df_raw = pd.read_csv(DATA_PATH)
    
    X, y = clean_and_prepare_data(df_raw, is_training=True)
    model, X_test, y_test = train_production_model(X, y)
    run_full_evaluation(model, X_test, y_test)
    run_live_simulation(DATA_PATH, num_batches=5)

if __name__ == "__main__":
    main()