import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import KNNImputer

def clean_and_prepare_data(df, is_training=True):
    if 'Time' in df.columns:
        df = df.drop(columns=['Time'])
    
    target = 'Pass/Fail'
    y = df[target].replace(-1, 0) if target in df.columns else None
    X = df.drop(columns=[target]) if target in df.columns else df.copy()
    
    if not os.path.exists('models'): os.makedirs('models')

    if is_training:
        print("Preprocessing:")

        null_thresh = 0.5
        X = X.dropna(thresh=X.shape[0] * (1 - null_thresh), axis=1)
        
        unique_counts = X.nunique()
        X = X.drop(columns=unique_counts[unique_counts <= 1].index)
        
 
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        
    
        joblib.dump(imputer, 'models/knn_imputer.pkl')
        joblib.dump(list(X.columns), 'models/feature_columns.pkl')
        
        return X_imputed, y
    else:
       
        features = joblib.load('models/feature_columns.pkl')
        imputer = joblib.load('models/knn_imputer.pkl')
       
        X = X[features]
        X_imputed = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
        return X_imputed, y