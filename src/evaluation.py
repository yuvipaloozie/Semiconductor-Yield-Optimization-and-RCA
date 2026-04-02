import os
import numpy as np
import pandas as pd
import shap
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import PartialDependenceDisplay, partial_dependence 

warnings.filterwarnings('ignore')

def simulate_intervention(model, X_test, y_test, feature, safe_target, threshold):
    probs_before = model.predict_proba(X_test)[:, 1]
    original_fails_mask = (probs_before >= threshold) & (y_test == 1)
    original_count = np.sum(original_fails_mask)

    if original_count == 0: return 0, 0

    bad_batches_fixed = X_test.copy()
    bad_batches_fixed.loc[original_fails_mask, feature] = safe_target
    
    probs_after = model.predict_proba(bad_batches_fixed)[:, 1]
    saved_count = np.sum(probs_after[original_fails_mask] < threshold)

    return original_count, saved_count

def run_full_evaluation(model, X_test, y_test):
    if not os.path.exists('reports'): os.makedirs('reports')
    if not os.path.exists('models'): os.makedirs('models')

    COST_FP = 500   
    COST_FN = 10000 
    
    y_probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = [(np.sum((y_probs >= t) & (y_test == 0)) * COST_FP) + 
             (np.sum((y_probs < t) & (y_test == 1)) * COST_FN) for t in thresholds]
    
    best_threshold = thresholds[np.argmin(costs)]
    print(f"Optimal Threshold Found: {best_threshold:.3f}")
    
    with open('models/threshold.txt', 'w') as f:
        f.write(str(best_threshold))

    # Confusion Matrix
    y_pred = (y_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f'Confusion Matrix (Threshold: {best_threshold:.3f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    print("Computing SHAP values:")
    
    X_test_summary = shap.kmeans(X_test, 5)
    predict_fn = lambda x: model.predict_proba(x)
    explainer = shap.KernelExplainer(predict_fn, X_test_summary)
    
    shap_vals = explainer.shap_values(X_test.iloc[:30, :])
    
    if isinstance(shap_vals, list):
        shap_val_target = shap_vals[1] 
    elif len(np.array(shap_vals).shape) == 3:
        shap_val_target = np.array(shap_vals)[:, :, 1] 
    else:
        shap_val_target = np.array(shap_vals) 
        
    mean_abs_shap = np.abs(shap_val_target).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
    
    top_features = X_test.columns[top_indices].tolist()
    print(f"Top 5 critical sensors identified: {top_features}")

    joblib.dump(top_features, 'models/top_features.pkl')

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_val_target, X_test.iloc[:30, :], max_display=5, show=False)
    plt.title('Top 5 Drivers of Batch Failure (SHAP)')
    plt.savefig('reports/shap_summary.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(18, 4)) 
    PartialDependenceDisplay.from_estimator(
        model, X_test, features=top_features, ax=ax, grid_resolution=50
    )
    plt.suptitle('Partial Dependence Risk Cliffs: Top 5 Sensors', y=1.05, fontsize=14)
    plt.savefig('reports/partial_dependence.png', bbox_inches='tight')
    plt.close()

    print("\n PDP Risk Cliffs")
    pdp_limits = {}
    
    for feature in top_features:
        pd_results = partial_dependence(model, X_test, [feature], kind="average", grid_resolution=100)
        grid_values = pd_results['grid_values'][0]
        risk_scores = pd_results['average'][0]

        # Find the "cliff" using the mathematical gradient
        gradient = np.gradient(risk_scores, grid_values)
        max_grad_idx = np.argmax(np.abs(gradient))
        limit_value = grid_values[max_grad_idx]

        # Use gradient sign to determine limit direction (Upper vs Lower bound)
        slope_at_limit = gradient[max_grad_idx]
        direction = "Upper" if slope_at_limit > 0 else "Lower"
        
        pdp_limits[feature] = {
            'limit': limit_value, 
            'direction': direction, 
            'avg': X_test[feature].mean()
        }
        print(f"Sensor {feature}: Limit = {limit_value:.3f} ({direction})")

    joblib.dump(pdp_limits, 'models/pdp_limits.pkl')

    print(f"\nIntervention Analysis: (Threshold: {best_threshold:.3f}) ---")
    print(f"{'Sensor':<10} | {'Action':<20} | {'Batches Saved':<15} | {'Success Rate':<10}")
    
    pass_indices = y_test[y_test == 0].index
    X_pass = X_test.loc[pass_indices]

    for feature in top_features:
        safe_target = X_pass[feature].median()
        total_fails, saved_fails = simulate_intervention(
            model, X_test, y_test, feature, safe_target, best_threshold
        )
        success_rate = (saved_fails / total_fails * 100) if total_fails > 0 else 0
        print(f"{feature:<10} | Set to {safe_target:<12.2f} | {saved_fails}/{total_fails}           | {success_rate:.1f}%")