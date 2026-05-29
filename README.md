# Credit Card Fraud Detection using FROID Framework 🚀

This repository implements a high-performance Fraud Detection system based on the **FROID** (Features Reduction and OutlIer Detection) framework, as proposed in the paper: *"Solving imbalanced learning with outlier detection and features reduction" (2023)*.

## 📌 Project Overview
The primary challenge in fraud detection is the **Extreme Class Imbalance** (e.g., fraud cases < 0.2%). Traditional methods like SMOTE often fail by creating noisy synthetic data. This project follows the **Representation Engineering** paradigm: instead of adding new samples, we enrich the feature space using unsupervised learning to make the minority class more separable.

## 🛠 Methodology: The FROID Pipeline
Our implementation enhances the original features through two main unsupervised streams:

1.  **Outlier Detection (OD) Features:**
    *   **Algorithms:** `IsolationForest` and `LocalOutlierFactor`.
    *   **Logic:** Fraudulent transactions often behave as outliers. We append "Outlier Scores" as new features to help the classifier identify anomalies.
2.  **Feature Reduction (FR) Features:**
    *   **Algorithm:** `PCA` (Principal Component Analysis).
    *   **Logic:** Projecting data into low-dimensional subspaces to capture hidden geometric structures and non-linear relationships.
3.  **Classification & Optimization:**
    *   **Model:** `LightGBM` with `class_weight='balanced'`.
    *   **Hyperparameter Tuning:** Automated optimization using **Optuna** to maximize **PR-AUC**.

## 📊 Key Results
The FROID framework, combined with Optuna tuning, showed significant improvement over the standard baseline.

| Model | Recall (Class 1) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: |
| Baseline (LightGBM) | 0.43 | 0.7095 | 0.1870 |
| Weighted Baseline | 0.87 | 0.9770 | 0.8864 |
| **Optimized FROID (Final)** | **0.84** | **0.9716** | **0.8935** |

> **Key Takeaway:** The Optimized FROID model achieved the highest **Precision-Recall AUC (0.8935)**, meaning it significantly reduced False Positives while maintaining high detection rates.

## 📂 Project Structure
```text
├── data/               # Raw and processed datasets
├── notebooks/          # Step-by-step implementation (Baseline & FROID)
├── src/                # Modular Python scripts for preprocessing and modeling
├── results/            # Saved models (.joblib) and performance plots
├── requirements.txt    # Project dependencies
└── README.md
