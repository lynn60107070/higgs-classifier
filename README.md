# Higgs Classification using Apache Spark

## DSAI4202 – Big Data Analysis with Machine Learning  
**Student:** Lynn Younes (60107070)

---

## Project Overview

This project builds a **scalable machine learning pipeline using Apache Spark** to classify particle collision events as:

- **Signal (1):** Higgs boson event  
- **Background (0):** Non-Higgs event  

The goal is to demonstrate **end-to-end big data processing and machine learning** on a large-scale dataset.

---

## What is the Higgs Boson?

The Higgs boson is a fundamental particle discovered in 2012 at CERN. It is associated with the Higgs field, which gives mass to other particles.

In this dataset:
- Each row represents a particle collision event  
- The task is to classify whether the event contains a Higgs boson signal  

---

## Dataset Description

- **Dataset:** HIGGS.csv  
- **Target:** `label` (binary classification)  
- **Total Features:** 28 numerical features  
  - **21 low-level features:** detector measurements  
  - **7 high-level features:** engineered physics variables  

---

## Technologies Used

- Apache Spark (PySpark / MLlib)
- Python (NumPy, Pandas)
- Scikit-learn
- Matplotlib, Seaborn

---

## Project Pipeline

1. Data ingestion (Spark DataFrame)
2. Data cleaning and preprocessing
3. Feature engineering
4. Exploratory Data Analysis (EDA)
5. Model training and comparison
6. Hyperparameter tuning
7. Model evaluation
8. Feature importance analysis
9. Final model testing
10. Overfitting analysis

---

## Exploratory Data Analysis (EDA)

### Key Findings

- Dataset is **fairly balanced**
- Most features are **right-skewed**
- Strong **overlap between classes**
- No single feature separates classes clearly
- Some features are **highly correlated**

### Feature Insights

- Strong features:
  - Mass features (`m_bb`, `m_wwbb`, `m_jjj`)
  - Momentum features (`jet_1_pt`, etc.)

- Weak features:
  - `phi` (uniform distribution)
  - `eta` (low impact)

---

## Models Compared

| Model                  | Description |
|------------------------|------------|
| Logistic Regression    | Baseline linear model |
| Decision Tree          | Simple nonlinear model |
| Random Forest          | Ensemble model |
| Gradient Boosted Trees | Boosting-based model |

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- **ROC AUC (main metric)**  

### Why ROC AUC?

- Evaluates performance across **all thresholds**
- Measures **class separation ability**
- Robust to class imbalance
- Provides overall model quality

---

## Model Performance (Validation)

| Model                    | Accuracy | F1     | Precision | Recall | ROC AUC |
|--------------------------|----------|--------|-----------|--------|---------|
| Gradient Boosted Trees   | 0.7273   | 0.7268 | 0.7267    | 0.7273 | 0.8051  |
| Random Forest            | 0.7099   | 0.7083 | 0.7092    | 0.7099 | 0.7830  |
| Logistic Regression      | 0.6459   | 0.6383 | 0.6459    | 0.6459 | 0.6877  |
| Decision Tree            | 0.6930   | 0.6920 | 0.6921    | 0.6930 | 0.6839  |

---

## Hyperparameter Tuning

### Tuned Model (Manual GBT)

| Model              | Accuracy | F1     | Precision | Recall | ROC AUC |
|--------------------|----------|--------|-----------|--------|---------|
| Tuned GBT (Manual) | 0.7320   | 0.7315 | 0.7315    | 0.7320 | 0.8113  |

### Why Manual Tuning?

- Faster than grid search  
- Suitable for large datasets  
- Reduces computational cost  
- Allows controlled parameter adjustments  

---

## Final Model Performance (Test Set)

| Model             | Accuracy | F1     | Precision | Recall | ROC AUC |
|-------------------|----------|--------|-----------|--------|---------|
| Final Model (GBT) | 0.7316   | 0.7313 | 0.7312    | 0.7316 | 0.8110  |

### Key Insight

- Validation and test scores are very close  
- Model shows **strong generalization**

---

## Feature Importance (Final Model)

### Most Important Features

- `m_bb`
- `m_wwbb`
- `m_jjj`
- `m_jlv`

### Other Important Features

- `jet_1_pt`
- `lepton_pT`
- `m_wbb`
- `m_jj`

### Low Importance

- `eta` features  
- `phi` features  
- `b_tag`  

---

## Overfitting Analysis

| Dataset     | ROC AUC |
|------------|--------|
| Train       | 0.8194 |
| Validation  | 0.8113 |
| Test        | 0.8110 |

### Interpretation

- Small gap → no overfitting  
- Consistent results → good generalization  

---

## Conclusion

- **Gradient Boosted Trees is the best model**
- Ensemble models outperform simpler models
- Feature interactions are critical
- Mass-based features dominate predictions
- Final model is **stable and scalable**

---

## Limitations

- Full cross-validation not feasible  
- Manual tuning used instead of full search  
- Training performed on sampled dataset  
- Some feature redundancy remains  

---

## Future Work

- Use advanced boosting models (XGBoost, LightGBM)
- Perform automated hyperparameter tuning
- Train on full dataset with distributed clusters
- Apply feature selection or PCA
- Optimize decision threshold

---

## How to Run

```bash
# Start Spark
pyspark

# Run notebook or script
```

---

## Project Structure
```bash
higgs-classifier/
├── data/ # Dataset and processed Spark files
├── models/ # Saved models and metrics
├── figures/ # Plots and visualizations
├── checkpoints/ # Intermediate checkpoints (optional)
├── notebook.ipynb # Main analysis notebook
├── README.md # Project documentation
```


---

## Author

**Lynn Younes**  
DSAI4202 – Information Retrieval