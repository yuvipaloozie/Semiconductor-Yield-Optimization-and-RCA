# Intelligent Process Control: Yield Optimization & Root Cause Analysis

![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-orange?style=for-the-badge)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-ff69b4?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-Manufacturing-lightgrey?style=for-the-badge)

## 1. Executive Summary
In semiconductor manufacturing, yield excursions (scrap batches) are incredibly costly. Traditional Statistical Process Control (SPC) often misses complex, multivariate interactions that lead to defects.

This project developed a machine learning pipeline to:
1.  **Predict Failures:** Identifying **48% of yield excursions** that were previously missed by standard controls.
2.  **Optimize Business Value:** Tuned the model based on financial impact (Cost of Scrap vs. Cost of Inspection) rather than just raw accuracy.
3.  **Define Control Limits:** Used "Virtual Metrology" to propose actionable changes to the Process Control Plan (PCP), specifically tighter limits on **Sensor 103**.

---

## 2. The Business Problem
* **The Data:** UCI SECOM Dataset (Semiconductor Manufacturing). 590 sensors, ~1500 batches.
* **The Challenge:**
    * **Extreme Class Imbalance:** Failures are rare (~6%), making standard models biased toward "Pass."
    * **High Dimensionality & Noise:** Hundreds of redundant or "dead" sensors.
    * **Cost Asymmetry:** A missed failure (False Negative) costs **$10,000**, while a false alarm (False Positive) costs only **$500**.

---

## 3. Solution Architecture

### Phase 1: Data Engineering ("The Sanitation Layer")
Raw sensor data is rarely model-ready. I implemented a robust preprocessing pipeline:
* **Variance Thresholding:** Removed 100+ "dead" sensors (zero variance).
* **Multicollinearity Filter:** Dropped redundant features ($r > 0.95$) to reduce noise.
* **KNN Imputation:** Used K-Nearest Neighbors to fill missing data, preserving the physical correlation structure between sensors (e.g., Temp/Pressure relationships).

### Phase 2: Cost-Sensitive Modeling
I trained an **XGBoost Classifier** specifically tuned for imbalance:
* **Class Weights:** Applied `scale_pos_weight` to heavily penalize missed failures.
* **Hyperparameter Tuning:** Used `RandomizedSearchCV` to optimize tree depth and learning rate.
* **Performance:**
    * **ROC-AUC:** Improved from 0.50 (Baseline) to **0.733** (Tuned).
    * **Recall:** The final model captures **48%** of all defects.

### Phase 3: Financial Optimization
Standard model thresholds (0.50) are suboptimal for manufacturing. I calculated the "Business Cost Curve" to find the sweet spot.

* **F1-Score Threshold:** 0.45 (Balances Precision/Recall statistically).
* **Minimum Cost Threshold:** **0.35** (Aggressive detection).
* **Decision:** We selected the **0.35** threshold. While this increases false alarms, it minimizes total financial loss by catching the most expensive scrap events.

![Business Cost Curve](YOUR_IMAGE_PATH_HERE/cost_curve.png)
*Figure 1: The "Sweet Spot" for the decision threshold that minimizes total business cost.*

---

## 4. Engineering Insights (Root Cause Analysis)

### The Primary Driver: Sensor 103
Using **SHAP (SHapley Additive exPlanations)**, we identified **Sensor 103** as the #1 predictor of failure.

![SHAP Summary Plot](YOUR_IMAGE_PATH_HERE/shap_summary.png)

### Virtual Metrology: Defining New Limits
To make this actionable, I used **Partial Dependence Plots (PDP)** to define the "Safe Operating Window" for Sensor 103.

* **Observation:** The process is stable when Sensor 103 reads below **-0.012**.
* **Risk Spike:** Failure probability doubles immediately when the value crosses **-0.009**.
* **Recommendation:** Tighten the Upper Control Limit (UCL) for Sensor 103 to **-0.012**.

![Sensor 103 PDP](YOUR_IMAGE_PATH_HERE/pdp_plot.png)
*Figure 2: Virtual Metrology showing the exact "Risk Cliff" for Sensor 103.*

---

## 5. Conclusion & Value Add
This project moves beyond "black box" prediction to provide transparent engineering solutions. By implementing the recommended control limit on Sensor 103 and using the cost-optimized detection model, the manufacturing process can significantly reduce scrap rates, translating to estimated savings of **$50,000 - $100,000 per year** (based on projected scrap reduction).

---

## 6. How to Run This Project
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/secom-yield-optimization.git](https://github.com/YOUR_USERNAME/secom-yield-optimization.git)
    ```
2.  **Install Requirements:**
    ```bash
    pip install pandas numpy xgboost shap scikit-learn matplotlib seaborn
    ```
3.  **Run the Notebook:**
    Open `SECOM_Yield_Analysis.ipynb` in Jupyter or Google Colab.
