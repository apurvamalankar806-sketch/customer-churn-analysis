# Customer Churn Analysis and Prediction

Exploratory analysis and predictive modelling of telecom customer churn in Python. The project identifies which customers are most likely to leave and compares three classifiers on how well they find them.

## Dataset
5043 telecom customers with contract, service, billing and tenure attributes. Approximately 30% of customers churned.

## Key Findings
- **Contract type:** month-to-month customers churned at 46%, compared with 13% for one-year and 3% for two-year contracts.
- **Internet service:** fiber optic customers churned at 42%, compared with 17% for DSL.
- **Tenure and charges:** shorter tenure and higher monthly charges are associated with higher churn.
- Tenure, total charges and monthly charges were the features the Random Forest relied on most. These are associations in the data and do not establish causes.

## Methodology
1. Cleaned the data: converted `TotalCharges` to numeric, removed incomplete rows and encoded the churn label.
2. Explored churn against monthly charges and tenure.
3. One-hot encoded categorical features and made a stratified 80/20 train-test split with a fixed random seed.
4. Trained three models and evaluated them on the held-out test set of 824 customers.

## Results
Approximately 70% of the test set did not churn, so a model predicting "no churn" for everyone would score about 70% accuracy. Models were therefore judged on precision, recall and F1 for the churn class.

| Model | Churner recall | Churner precision | Churner F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.57 | 0.65 | 0.61 | 0.78 |
| Logistic Regression (balanced) | 0.78 | 0.53 | 0.63 | 0.73 |
| Random Forest (balanced) | 0.72 | 0.60 | 0.66 | 0.77 |

All three models reached a ROC-AUC of about 0.83, so they rank customers almost equally well. The differences come from where each model sets its cut-off for flagging a churner.

**Model choice.** If a lost customer costs more than a retention offer, the balanced logistic regression is preferable because it identifies the most churners. If both errors cost about the same, the Random Forest gives the best balance.

## Limitations
- Single dataset and a single train-test split. With 247 churners in the test set, small differences between models may be due to chance.
- The analysis shows which customers are at risk, not why they leave.
- The decision threshold was not tuned and is a natural next step.

## Tech Stack
Python, Pandas, Matplotlib, Seaborn, Scikit-learn

## Repository Contents
- `churn_analysis_v2.ipynb`: full analysis and modelling
- `telco_churn.csv`: dataset
