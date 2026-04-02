import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def train_production_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Balancing training data using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=42, eval_metric='logloss'
    )
    
    print("Training XGBoost...")
    model.fit(X_train_res, y_train_res)
    joblib.dump(model, 'models/secom_v1_model.pkl')
    
    return model, X_test, y_test