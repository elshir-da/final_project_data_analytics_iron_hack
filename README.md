# Credit Card Default Prediction

## Project Overview

This project develops an end-to-end machine learning workflow to predict whether a credit card customer will default on their payment in the following month. It uses the **Default of Credit Card Clients** dataset and evaluates predictive performance alongside hypothetical banking risk-management scenarios.

The goal is to support earlier identification of customers at risk of default while recognizing the trade-off between missed defaults and unnecessary reviews.

> **Project status:** The machine learning analysis and exploratory business simulations are complete. Data engineering integration and production deployment are proposed future extensions, not completed functionality.

## Business Problem

Credit card default is a challenge for financial institutions. A model that misses too many defaulting customers may fail to identify important risks, while a model that flags too many non-defaulting customers may generate unnecessary review costs.

This project asks:

- Can customer repayment, billing, and demographic data help predict next-month default?
- How should precision and recall be balanced for risk identification?
- How could model scores support hypothetical review prioritization and risk segmentation?

## Dataset

**Source:** UCI Machine Learning Repository — *Default of Credit Card Clients*  
**Dataset link:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

- **Observations:** 30,000 customers
- **Original columns:** 25, including customer ID and target
- **Target:** `default_next_month` (1 = default; 0 = no default)
- **Class distribution:** 6,636 defaults (22.12%) and 23,364 non-defaults (77.88%)

Variables cover credit limits, demographic characteristics, repayment status, bill amounts, and previous payment amounts. The dataset is historical; findings should not be assumed to generalize to today's banking populations.

## Project Workflow

1. **Data loading and inspection:** Review the dataset structure, target balance, and variable definitions.
2. **Data cleaning:** Prepare a consistent dataset for analysis.
3. **Exploratory data analysis (EDA):** Examine distributions and relationships with default.
4. **Feature engineering and preprocessing:** Create behavioral features and apply appropriate preprocessing inside scikit-learn pipelines.
5. **Model development:** Establish SGD, Random Forest, and XGBoost baselines; tune XGBoost using cross-validation.
6. **Final evaluation:** Evaluate classification metrics, confusion matrix, feature importance, and SHAP explanations.
7. **Business simulations:** Explore hypothetical error costs, risk review capacity, risk segments, and intervention sensitivity.
8. **Conclusion and future work:** Discuss practical implications, limitations, and next steps.

### Feature Engineering

Six additional features were developed:

- `avg_bill_amount`
- `avg_payment_amount`
- `payment_to_bill_ratio`
- `credit_utilization`
- `max_payment_delay`
- `months_with_delay`

An `age_group` variable was used for exploratory analysis only and excluded from model training.

### Modeling Approach

- **Train/test split:** Stratified 80/20 split (`random_state=1`)
- **Preprocessing:** Imputation, scaling, and categorical encoding in a pipeline
- **Models considered:** SGD classifier, Random Forest, and XGBoost
- **Primary tuning metric:** Cross-validation F1-score
- **Selected model:** Weighted XGBoost with `scale_pos_weight = 2.5`

F1-score was selected to balance detecting actual defaults (recall) against limiting false alarms (precision). Alternative class weights were explored for comparison: `1.0` favored cross-validation accuracy/precision, while `4.0` favored recall. The final weight of `2.5` achieved the highest cross-validation F1-score among the tested values.

## Results

### Selected Weighted XGBoost — Test Set

| Metric | Result |
|---|---:|
| Accuracy | 0.7897 |
| Precision | 0.5224 |
| Recall | 0.5712 |
| F1-score | 0.5457 |
| ROC-AUC | 0.7818 |

**Confusion matrix (6,000 test customers):**

| Actual / Predicted | No Default | Default |
|---|---:|---:|
| No Default | 3,980 (TN) | 693 (FP) |
| Default | 569 (FN) | 758 (TP) |

Compared with the random-search XGBoost configuration, the selected weighted model detected **263 additional actual defaults**, while generating **448 additional false positives** on the same test set.

These results illustrate a trade-off rather than an unqualified improvement across all metrics.

### Model Interpretability

Feature importance and SHAP analyses highlight repayment delays and repayment history as important model signals. The analyses describe model behavior and associations; they do not establish causal relationships.

## Exploratory Business Simulations

All figures in this section are **illustrative scenario calculations**, not observed bank costs, actual interventions, or realized savings.

### Prediction Error Cost Scenario

Using hypothetical costs of **€1,000 per false negative** and **€50 per false positive**:

| Model | Illustrative Error Cost |
|---|---:|
| Random-search XGBoost | €844,250 |
| Weighted XGBoost (`scale_pos_weight = 2.5`) | €603,650 |

These assumptions demonstrate how the relative costs of missed defaults and false alarms can change the preferred model.

### Risk Review Capacity

Ranking test customers by predicted risk score, the top **500** contained **361 actual defaults**, corresponding to **72.2% precision** and **27.20% recall** for that review capacity. This is a retrospective ranking analysis, not evidence that defaults would be prevented.

### Risk Segmentation

An illustrative segmentation of the 6,000 test customers used model-score thresholds:

| Segment | Score Range | Customers | Observed Default Rate |
|---|---|---:|---:|
| Low | < 0.30 | 3,063 | 9.37% |
| Medium | 0.30–<0.60 | 1,902 | 21.77% |
| High | ≥ 0.60 | 1,035 | 60.48% |

These thresholds are exploratory, and the class-weighted model scores have **not** been established as calibrated default probabilities.

### Hypothetical Intervention Strategies

| Strategy | Modeled Cost |
|---|---:|
| A — No interventions | €1,327,000 |
| B — Manual review above score 0.50 | €1,233,440 |
| C — High-risk manual review plus medium-risk reminders | €1,226,304 |

Under the stated assumptions, Strategy C is **€100,696 lower in modeled cost than Strategy A**. Sensitivity analysis favored Strategy C in three of four tested scenarios; when reminders had no effect, Strategy B performed better.

These differences **must not** be presented as verified financial savings or prevented defaults.

## Reproducing the Analysis

### Requirements

- Python 3.x
- Jupyter Notebook or JupyterLab
- Common Python data science packages, including `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, and `shap`

Install the required packages in your own environment, for example:

```bash
pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost shap
```

### Run the Notebook

1. Download the dataset from the UCI repository.
2. Place the dataset in the location expected by the notebook, or update the input file path.
3. Open the project's main Jupyter notebook.
4. Select **Restart Kernel and Run All Cells** to reproduce the workflow.

The repository's exact notebook name and data paths may vary; verify them before running. The dataset and environment dependencies should be documented consistently with the committed files.

## Limitations

- The dataset is historical and may not represent current customers or economic conditions.
- Default is an imbalanced outcome, making accuracy insufficient as a stand-alone selection metric.
- Repeated exploratory comparisons on the same test set limit the independence of the reported evaluation.
- Class-weighted risk scores are not necessarily calibrated probabilities.
- Business simulations rely on assumed losses, operational costs, and intervention effectiveness.
- Fairness, independent external validation, operational monitoring, and human oversight require further work before deployment.

## Future Work

- Validate on newer, representative, independent datasets.
- Evaluate probability calibration, subgroup fairness, and model drift.
- Validate business assumptions with real operational evidence.
- Develop the planned data engineering workflow with external API/web data, MySQL/SQL, BigQuery, and a Flask API.
- Establish appropriate human review and governance for consequential decisions.

## Project Scope and Disclaimer

This project is an **educational proof of concept**, not a deployable credit decision system. Predictions, rankings, and hypothetical financial scenarios must not be used as the sole basis for real-world credit decisions.

## Acknowledgment

Dataset: *Default of Credit Card Clients*, UCI Machine Learning Repository. Original study by Yeh and Lien (2009).
