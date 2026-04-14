# Higgs Classification using Apache Spark

## DSAI4202 – Big Data Analysis with Machine Learning  
**Student:** Lynn Younes (60107070)

---

## Project Overview

This project builds a scalable machine learning pipeline using Apache Spark to classify particle collision events as:

- Signal (1): Higgs boson event  
- Background (0): Non-Higgs event  

The goal is to demonstrate end-to-end big data processing and machine learning on a large-scale dataset.

---

## What is the Higgs Boson?

The Higgs boson is a fundamental particle discovered in 2012 at CERN. It is associated with the Higgs field, which gives mass to other particles.

Identifying Higgs events is important because:
- It validates the Standard Model of particle physics  
- Helps scientists understand how mass is generated  
- Enables discovery of new physics beyond current theories  

In this dataset:
- Each row represents a particle collision event  
- The task is to classify whether the event contains a Higgs boson signal  

---

## Dataset Description

- Dataset: HIGGS.csv  
- Target: `label` (binary classification)  
- Total Features: 28 numerical features  
  - 21 low-level features: detector measurements  
  - 7 high-level features: engineered physics variables  

Low-level features are directly measured from detectors, while high-level features are derived by physicists to capture complex particle interactions.

---

## Technologies Used

- Apache Spark (PySpark / MLlib)
- Python (NumPy, Pandas)
- Scikit-learn
- Matplotlib, Seaborn

Apache Spark was chosen over Pandas because:
- It supports distributed computing across multiple cores/machines  
- It handles datasets larger than memory  
- It uses lazy evaluation, optimizing execution only when needed  

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

## Pipeline Overview (Simplified)

Raw CSV
→ Spark DataFrame
→ Cleaning
→ Feature Engineering
→ VectorAssembler
→ Model Training (GBT)
→ Evaluation (ROC AUC)

---

## Data Ingestion Notes

- Column names were manually defined because the dataset has no header
- Large CSV files can cause issues such as:
  - Slow parsing
  - High memory usage
  - Schema inference errors

---

## Data Cleaning and Preprocessing

- Removed duplicate rows (~278k duplicates)
- Dropped missing values (none present, but ensured robustness)
- Converted label to double type (required by Spark ML models)

Duplicates may exist due to:
- Data generation process in simulations
- Data merging or replication

Duplicates were removed to avoid biasing the model.

If missing values existed, imputation could be used, but dropping was safe here due to data completeness.

---

## Exploratory Data Analysis (EDA)

### Key Findings

- Dataset is fairly balanced
- Most features are right-skewed
- Strong overlap between classes
- No single feature separates classes clearly
- Some features are highly correlated

### Deeper Insights

- Right-skewness indicates most values are small with extreme high-value outliers
- Multicollinearity (feature correlation) matters because:
  - It can affect linear models
  - Tree models are more robust to it

Feature vs label analysis showed:
- Slight shifts between classes
- Heavy overlap → requires combining multiple features

Most features are weakly correlated with the label because:
- Higgs detection depends on complex interactions, not single variables

---

## Feature Engineering

### Engineered Features

- `feat_sum`: total signal strength across features
- `feat_l2`: magnitude of feature vector

These help the model by:
- Capturing overall energy/scale of events
- Providing aggregated signals that may not be visible in individual features

### Transformations

- VectorAssembler: required to convert columns into a feature vector
- StandardScaler: used for models sensitive to scale (e.g., Logistic Regression)

### PCA

- Not used because tree-based models handle correlated features well

PCA would be useful if:
- Using linear models
- Reducing dimensionality for speed
- Handling extremely high-dimensional data

---

## Models Compared

| Model                  | Description |
|------------------------|------------|
| Logistic Regression    | Baseline linear model |
| Decision Tree          | Simple nonlinear model |
| Random Forest          | Ensemble model |
| Gradient Boosted Trees | Boosting-based model |

### Model Insights

- Decision Trees:
  - Easy to interpret but prone to overfitting

- Random Forest:
  - Reduces overfitting using multiple trees (bagging)

- Gradient Boosted Trees:
  - Builds trees sequentially to correct errors
  - More powerful but more computationally expensive

Logistic Regression performed poorly because:
- Data relationships are highly nonlinear

---

## Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC AUC (main metric)  

### Why ROC AUC?

- Evaluates performance across all thresholds
- Measures ranking ability of the model
- Robust to class imbalance

ROC AUC measures how well the model separates classes:
- 0.5 → random guessing
- 1.0 → perfect classification
- ~0.81 → strong separation ability

### Metric Trade-offs

- Accuracy can be misleading with imbalance
- Precision = correctness of positive predictions
- Recall = ability to find all positives

Use cases:
- Precision → avoid false positives
- Recall → avoid false negatives

---

## Hyperparameter Tuning

### Key Parameters

- maxDepth → controls model complexity
- maxIter → number of boosting iterations
- stepSize → learning rate (controls update size)

Manual tuning was used because:
- Full grid search is expensive on large datasets
- Faster and more practical for big data

Cross-validation:
- Improves robustness
- Limited in big data due to computational cost

---

## Final Model Performance (Test Set)

| Model             | Accuracy | F1     | Precision | Recall | ROC AUC |
|-------------------|----------|--------|-----------|--------|---------|
| Final Model (GBT) | 0.7316   | 0.7313 | 0.7312    | 0.7316 | 0.8110  |

### Key Insight

- Validation and test scores are very close
- Small gap indicates:
  - No overfitting
  - Good generalization

---

## Feature Importance (Final Model)

### Most Important Features

- m_bb
- m_wwbb
- m_jjj
- m_jlv

These are dominant because:
- They represent invariant mass relationships critical in physics

Feature importance helps:
- Understand model behavior
- Identify key drivers of predictions

Low-importance features were not removed to preserve potential interactions.

---

## Overfitting Analysis

| Dataset     | ROC AUC |
|------------|--------|
| Train       | 0.8194 |
| Validation  | 0.8113 |
| Test        | 0.8110 |

### Interpretation

- Small gap → no overfitting
- Balanced performance → good bias-variance tradeoff

Overfitting is detected when:
- Training performance >> test performance

---

## Spark Concepts

- Distributed computing: processing data across multiple nodes
- Lazy evaluation: Spark delays execution until an action is triggered
- Spark MLlib vs Scikit-learn:
  - Spark → scalable, distributed
  - Scikit-learn → in-memory, single machine

Limitations of local Spark:
- Limited memory
- No true cluster scaling

---

## Key Technical Decisions

| Decision | Reason |
|--------|------|
| Used Apache Spark | Handles large-scale data (11M rows) with distributed computing |
| Used ROC AUC as main metric | Threshold-independent and robust to imbalance |
| Chose Gradient Boosted Trees | Best performance and captures nonlinear relationships |
| Did not use PCA | Tree models handle correlated features well |
| Used sampling for training | Reduced computational cost while preserving distribution |
| Used manual tuning | More efficient than full grid search for large data |

---

## Challenges and Solutions

- Large dataset size
  → Solution: Used Spark + sampling

- Computational constraints
  → Solution: Limited grid search and used manual tuning

- Feature overlap between classes
  → Solution: Used ensemble models to capture complex patterns

- Correlated features
  → Solution: Used tree-based models instead of PCA

---

## Why This Problem is Challenging

- No single feature clearly separates classes
- Strong overlap between signal and background
- Many features are weak predictors individually
- Requires combining multiple features nonlinearly

---

## If More Computational Resources Were Available

- Train on full 11M dataset instead of sample
- Perform full cross-validation with larger parameter grid
- Use distributed cluster (Spark cluster)
- Try advanced models like XGBoost / LightGBM

---


## Conclusion

- Gradient Boosted Trees is the best model
- Ensemble models outperform simpler models
- Feature interactions are critical
- Mass-based features dominate predictions
- Final model is stable and scalable

---

## Limitations

- Training performed on sampled dataset
- Limited hyperparameter tuning
- No automated optimization
- Feature redundancy remains

---

## Future Work

- Use XGBoost or LightGBM (better optimization, faster training)
- Perform automated tuning (grid/random/Bayesian)
- Train on full dataset using cluster
- Apply feature selection
- Optimize classification threshold

If dataset becomes imbalanced:
- Use resampling (SMOTE, undersampling)
- Adjust class weights

---

## Deployment Considerations

- Model can be deployed via Spark pipeline
- Threshold selection depends on use-case trade-off:
  - Higher precision vs higher recall

Trade-offs:
- Complex models → better performance, lower interpretability
- Simpler models → easier to explain

---

## Communication

To non-technical audience:
- “We built a system that can identify rare particle events with high accuracy using patterns in large-scale data.”

---

## Key Takeaway

No single feature detects the Higgs boson. Accurate classification requires combining multiple weak signals using powerful ensemble models.

---

## Author

Lynn Younes  
DSAI4202 – Information Retrieval