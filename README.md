# customer-churn-prediction 🐍3.14 🐼 📈 📉

Executive Summary:

  I have created a machine learning engine for predicting customer churn that compares 2 different models, logistic regression & random forest. Logistic regression proved to be the better model with a higher predictive score. The churn rate in the dataset was ~26%. The model answers the question: "What customer attributes affect churn, both positively and negatively?" A business would be able to use the segments listed in the key findings below.

Business Problem:

  - Why does churn matter? (cost of acquisition vs retention)
  - What is the churn rate in this dataset? (26%)
  - What question is the model answering?
  - What would a business do with the output? 

Dataset:

  - Source: IBM Telco Customer Churn dataset via Kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn 
  - Size: 7,032 customers, 21 features
  - Target variable: Churn (Yes/No → 1/0)
  - Notable features: tenure, contract type, monthly charges, internet service
  - Data quality note: TotalCharges were blank in 11 rows. These were new customers with 0 tenure. These null values were replaced with whitespace.

Methodology:

  Exploratory data analysis reveled that the churn rate was at ~26%. The key segments driving this were contract type, type of internet service and monthly charges. Total charges were highly correlated with tenure and were adjusted accordingly. I used logistic regression as a baseline and random forest as a comparative model. The model was evaluated using ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) to distinguish between the 2 models. The threshold tuning was adjusted from 0.5 to 0.38 to prioritize recall. Please see the Churn Rate by Key Segments chart for the segments that had the most impact.

Key Findings by Segment:

- Month-to-month customers churn at 43% vs 3% for two-year contracts — a 14x difference. Contract type is the single most actionable lever available
- Customers in their first 12 months churn at 48%, dropping to under 15% by month 25. The first year is the highest-risk window by a significant margin
- Fiber optic customers churn at 42% vs 19% for DSL users, suggesting either a pricing or service quality issue specific to that product line
- The highest-risk segment — month-to-month, fiber optic, under 12 months — churns at approximately 70%. This group represents the clearest intervention target
- Tenure, monthly charges, and contract type are the top 3 predictive features 

Results: 


  Logistic Regression:

  - ROC-AUC - 0.851
  - Precision (churn) - 0.67
  - Recall (churn) - 0.53
  - F1 (churn) - 0.59
  
  Random Forest:
  
  - ROC-AUC - 0.846
  - Precision (churn) - 0.71
  - Recall (churn) - 0.49
  - F1 (churn) - 0.58

  Logistic Regression performed slightly better given the higher ROC-AUC as well as the F1, therefore, we used it as the baseline. The 0.5 classification threshold was adjusted to 0.38 to increase recall on
the minority class, accepting more false positives in exchange for catching more true churners — appropriate given that the cost of missing a churner
exceeds the cost of a wasted retention call, in this case that was characterized by the F1 statistic.

Limitations & Future Work:

Limitations

  - Single snapshot dataset — no temporal dimension or seasonality
  - TotalCharges and tenure are highly correlated
  - Model trained on one company's data — may not generalise

Future Work

  - SHAP values for individual-level explainability
  - Survival analysis to predict when a customer will churn, not just if
  - XGBoost or LightGBM comparison
  - Deploy as a simple Flask or Streamlit app

Tech Stack:

  - Python 3.14
  - pandas — data manipulation
  - numpy — numerical operations
  - scikit-learn — modelling and evaluation
  - matplotlib / seaborn — visualisation

About the Author, Nick Morrelli:

15+ years of experience in marketing analytics and business intelligence with 10+ years of management experience, transitioning into data science with a focus on ML applications for marketing and customer analytics. Open to senior data science management or analyst roles.
