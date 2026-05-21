# LLM-Powered Financial Fraud Detection System

## Overview

This project builds a machine learning pipeline for detecting fraudulent credit card transactions in a highly imbalanced financial dataset. It combines exploratory data analysis, feature engineering, model training, threshold-aware evaluation, SHAP explainability, and an offline LLM-style explanation layer that translates model outputs into readable fraud investigation summaries.

The project is intentionally honest about its current scope: the completed work is notebook-based and offline. It does not yet include a deployed application, API, RAG system, LangChain workflow, Docker setup, or live demo.

## Why This Project Matters

Financial fraud detection is a high-stakes classification problem where fraudulent transactions are rare but costly. In this setting, a model can achieve very high accuracy by mostly predicting the majority class, while still missing the cases that matter most.

This project focuses on the more practical fraud detection questions:

- How well does the model identify fraudulent transactions?
- How many flagged transactions are actually fraudulent?
- Which features are driving model decisions?
- How can model outputs be translated into analyst-friendly explanations?

## Repository Structure

```text
.
|-- EDA.ipynb
|-- 02_modeling.ipynb
|-- 02_modeling_executed.ipynb
|-- 03_explainability_shap.ipynb
|-- 03_explainability_shap_executed.ipynb
|-- 04_llm_fraud_explanation_layer.ipynb
|-- 04_llm_fraud_explanation_layer_executed.ipynb
|-- LLM-powered Financial Fraud Detection System.docx
|-- data/
|-- outputs/
|-- .gitignore
`-- README.md
```

### Completed Stages

| Stage | Notebook | Status |
|---|---|---|
| Exploratory data analysis and cleaning | `EDA.ipynb` | Complete |
| Modeling and evaluation | `02_modeling.ipynb`, `02_modeling_executed.ipynb` | Complete |
| SHAP explainability | `03_explainability_shap.ipynb`, `03_explainability_shap_executed.ipynb` | Complete |
| Offline LLM-style explanation layer | `04_llm_fraud_explanation_layer.ipynb`, `04_llm_fraud_explanation_layer_executed.ipynb` | Complete |

## Dataset Description

The project uses a credit card fraud detection dataset with anonymized transaction features.

Key columns:

- `V1` through `V28`: anonymized PCA transaction behavior features
- `Time`: transaction timestamp relative to the dataset timeline
- `Amount`: transaction amount
- `Class`: target label
  - `0`: legitimate transaction
  - `1`: fraudulent transaction

The dataset is highly imbalanced, which is typical for fraud detection problems.

Data files are intentionally ignored and are not pushed to GitHub.

## Feature Engineering

The project includes engineered features designed to improve transaction pattern detection:

- `LogAmount`: log-transformed transaction amount
- `IsZeroAmount`: indicator for zero-amount transactions
- `Day`: derived time-based feature
- `HourSin`: cyclical hour encoding using sine transformation
- `HourCos`: cyclical hour encoding using cosine transformation
- `ExtremeFlag` features: tested during experimentation

The final best-performing model version did not use the `ExtremeFlag` features.

## Modeling Approach

The modeling workflow was designed around fraud detection best practices for imbalanced classification.

Key steps:

- Used a stratified train-test split
- Kept the test set imbalanced and untouched
- Applied SMOTE only to the training data
- Evaluated models using fraud-relevant metrics rather than relying on accuracy
- Compared model performance with and without tested engineered features
- Assessed threshold behavior for practical fraud alerting

Models tested:

- Logistic Regression
- Random Forest
- HistGradientBoostingClassifier

Main evaluation metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC / Average Precision

## Best Model Results

The best-performing model was the Random Forest model, Version B, without `ExtremeFlag` features.

| Model | Version | Threshold | Precision | Recall | F1-score | ROC-AUC | PR-AUC / Average Precision |
|---|---|---:|---:|---:|---:|---:|---:|
| Random Forest | Without ExtremeFlags | 0.50 | 0.924 | 0.768 | 0.839 | 0.986 | 0.816 |

## Why Accuracy Is Not the Main Metric

Accuracy is not emphasized because credit card fraud detection is a highly imbalanced classification problem. When fraudulent transactions represent only a small portion of the dataset, a model can appear accurate by predicting most transactions as legitimate.

For this reason, the project focuses on:

- **Precision**, to measure how reliable fraud alerts are
- **Recall**, to measure how many fraudulent transactions are caught
- **F1-score**, to balance precision and recall
- **PR-AUC / Average Precision**, to evaluate performance under class imbalance
- **ROC-AUC**, to assess overall ranking ability

## Threshold Tuning

The selected best model uses a threshold of `0.50`. Threshold tuning is important because fraud detection systems often need to balance competing business goals:

- Higher recall may catch more fraud but increase false positives
- Higher precision may reduce false alerts but miss more fraud

The final threshold should depend on investigation capacity, false positive cost, fraud loss exposure, and business risk tolerance.

## SHAP Explainability

SHAP was used to provide model-grounded explanations for fraud predictions. This helps identify which features contributed most strongly to model decisions.

Top SHAP features:

1. `V14`
2. `V12`
3. `V4`
4. `V3`
5. `V10`
6. `V11`
7. `V17`
8. `V16`
9. `V7`
10. `V1`

These features are anonymized PCA components. They should not be assigned invented real-world meanings. For example, `V14` should not be described as a specific customer behavior or transaction attribute unless that mapping is available from the original data source.

SHAP explanations in this project are used as model-grounded signals, not as causal claims.

## Offline LLM-Style Fraud Explanation Layer

The project includes an offline, rule-based prototype that converts model outputs and SHAP drivers into human-readable fraud investigation summaries.

This layer:

- Does not call the OpenAI API
- Does not call any paid API
- Does not use a live LLM service
- Converts model predictions into structured narrative summaries

Generated explanation components include:

- Risk summary
- Key model drivers
- Suggested investigation action
- Confidence caveat

This prototype demonstrates how model outputs could be packaged for fraud analysts while keeping the implementation fully offline and reproducible.

## Key Takeaways

- Fraud detection should be evaluated with imbalance-aware metrics, not accuracy alone.
- Keeping the test set imbalanced and untouched gives a more realistic estimate of model behavior.
- SMOTE should be applied only to the training data to avoid data leakage.
- Random Forest performed best among the tested models.
- SHAP adds transparency by identifying model drivers, but anonymized PCA features should not be over-interpreted.
- A lightweight explanation layer can make model results more usable for fraud investigation workflows.

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- SHAP
- Matplotlib
- Seaborn

## How to Run the Project

1. Clone the repository.

   ```bash
   git clone <repository-url>
   cd LLM-Financial-Fraud-Detection
   ```

2. Add the dataset locally.

   Place the credit card fraud detection dataset in the local `data/` directory. Data files are ignored by Git and are not included in this repository.

3. Install the required Python packages.

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn jupyter
   ```

4. Open Jupyter Notebook.

   ```bash
   jupyter notebook
   ```

5. Run the notebooks in order.

   ```text
   EDA.ipynb
   02_modeling.ipynb
   03_explainability_shap.ipynb
   04_llm_fraud_explanation_layer.ipynb
   ```

Executed notebook versions are included for review where available.

## Future Improvements

Planned enhancements may include:

- Building an interactive dashboard with Streamlit
- Creating a FastAPI endpoint for model inference
- Adding Docker support for reproducible environments
- Experimenting with LangChain for structured explanation workflows
- Exploring RAG for fraud investigation knowledge support
- Adding model monitoring and drift detection
- Improving threshold selection with business cost assumptions
- Creating a deployed demo once the application layer is implemented

These items are future improvements and are not claimed as completed in the current repository.

## Author

**Nazish Atta**  
MS Data Science Student  
George Washington University
