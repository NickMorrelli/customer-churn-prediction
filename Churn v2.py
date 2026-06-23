# Load and Explore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Step 1: Load
df = pd.read_csv('Datasets/Telco Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Rename tenure to Tenure
df.rename(columns={'tenure': 'Tenure'}, inplace=True)

# ── Step 2: First look 
print("Shape:", df.shape)          # (7043, 21) — rows, columns
print("\nColumn names:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:", 
      df.head())

# ── Step 3: Check for missing values 
print("Null counts:\n", df.isnull().sum())

# ── Step 4: Summary statistics for numeric columns 
print(df.describe())
# No: 0.734 (73.4%)
# Yes: 0.266 (26.6%)
# Imbalanced — a "predict nobody churns" model gets 73% accuracy for free.
# This is why we'll use ROC-AUC, not accuracy, to evaluate.

# ── Step 5: Understand the target variable first 
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(3))
# No: 0.734 (73.4%)
# Yes: 0.266 (26.6%)
# Imbalanced — a "predict nobody churns" model gets 73% accuracy for free.
# This is why we'll use ROC-AUC, not accuracy, to evaluate.

# Visualise it
df['Churn'].value_counts().plot(kind='bar', color=['steelblue', 'coral'])
plt.title('Churn Distribution')
plt.xlabel('Churned'); plt.ylabel('Count')
plt.xticks(rotation=0); plt.tight_layout(); plt.show()

# ── Step 6: Spot the hidden data quality issue 
print(df['TotalCharges'].dtype)          # object — should be float
print(df['TotalCharges'].unique()[:10])  # you'll see ' ' (whitespace) entries

# How many bad rows?
bad_rows = df[df['TotalCharges'].str.strip() == '']
print(f"Whitespace rows in TotalCharges: {len(bad_rows)}")  # 11 rows
# These are new customers with 0 tenure — TotalCharges was left blank.
# We'll fix this in the cleaning step.

# ── Step 7: EDA — churn rate by key categorical variables
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

cat_cols = ['Contract', 'InternetService', 'PaymentMethod']

for ax, col in zip(axes, cat_cols):
    churn_rate = df.groupby(col)['Churn'].apply(
        lambda x: (x == 'Yes').mean()
    ).sort_values(ascending=False)
    churn_rate.plot(kind='bar', ax=ax, color='coral')
    ax.set_title(f'Churn Rate By {col}')
    ax.set_ylabel('Churn Rate')
    ax.set_ylim(0, 0.6)
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout(); plt.show()
# We'll immediately see: month-to-month contracts churn at ~43%,
# two-year contracts at ~3%. Contract type will likely be your strongest feature.

# ── Step 8: EDA — churn vs numeric variables 
df['TotalCharges_clean'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges_clean']

for ax, col in zip(axes, num_cols):
    df[df['Churn'] == 'No'][col].hist(ax=ax, alpha=0.6, label='Stayed',
                                       bins=30, color='steelblue')
    df[df['Churn'] == 'Yes'][col].hist(ax=ax, alpha=0.6, label='Churned',
                                        bins=30, color='coral')
    ax.set_title(col); ax.legend()

plt.tight_layout(); plt.show()
# Key insight: churners cluster at low tenure (new customers leaving early)
# and high monthly charges. Long-tenure customers rarely churn.

# ── Step 9: Correlation heatmap 
# Numeric columns only
numeric_df = df[['Tenure', 'MonthlyCharges', 'TotalCharges_clean']].copy()
numeric_df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)

sns.heatmap(numeric_df.corr().round(2), annot=True, cmap='coolwarm',
            vmin=-1, vmax=1, square=True)
plt.title('Correlation Matrix'); plt.tight_layout(); plt.show()
# tenure correlates negatively with churn (longer tenure = less churn)
# MonthlyCharges correlates positively (higher charges = more churn)
# TotalCharges is highly correlated with tenure — worth noting for modelling

# ── Step 10: Summary of what we found 
print("=== EDA Summary ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Churn rate: {(df['Churn']=='Yes').mean():.1%}")
print(f"Data quality issue: 11 rows with blank TotalCharges")
print(f"\nStrongest signals spotted:")
print(f"  Contract:       month-to-month churn ~43%, two-year ~3%")
print(f"  Tenure:         churners median tenure much lower than stayers")
print(f"  MonthlyCharges: churners pay more on average")


# Clean and Encode

# ── Step 1: Fix TotalCharges (whitespace instead of NaN)
print("Before fix:", df['TotalCharges'].dtype)          # object
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("Nulls created:", df['TotalCharges'].isna().sum()) # should be 11

df.dropna(inplace=True)
print("Rows after dropping nulls:", len(df))            # 7032

# ── Step 2: Drop the ID column ────────────────────────────────────────────────
df.drop('customerID', axis=1, inplace=True)

# ── Step 3: Convert target to 0/1 ────────────────────────────────────────────
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print("\nChurn distribution:\n", df['Churn'].value_counts())

# ── Step 4: Inspect what's categorical before encoding ───────────────────────
cat_cols = df.select_dtypes(include='object').columns.tolist()
print("\nCategorical columns:", cat_cols)
# ['gender','Partner','Dependents','PhoneService','MultipleLines',
#  'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
#  'TechSupport','StreamingTV','StreamingMovies','Contract',
#  'PaperlessBilling','PaymentMethod']

# Spot-check unique values in a couple — good habit before blindly encoding
print(df['Contract'].value_counts())
print(df['InternetService'].value_counts())

# ── Step 5: One-hot encode all categoricals in one shot ──────────────────────
df = pd.get_dummies(df, drop_first=True)
print("\nShape after encoding:", df.shape)   # (7032, 30) approx
print(df.dtypes.value_counts())             # should be all bool/uint8/int64/float64

# ── Step 6: Sanity check — no object columns remain ──────────────────────────
remaining_objects = df.select_dtypes(include='object').columns.tolist()
print("\nObject columns remaining (should be []):", remaining_objects)

# ── Step 7: Peek at what get_dummies created ─────────────────────────────────
print("\nSample encoded columns:")
print([c for c in df.columns if 'Contract' in c])
# ['Contract_One year', 'Contract_Two year']
# 'Month-to-month' was dropped — it's the implicit baseline

print("\nFinal dataframe shape:", df.shape)
print(df.head(3))

# Split into train and test sets

from sklearn.model_selection import train_test_split

# Separate features from target
X = df.drop('Churn', axis=1)   # everything except what we're predicting
y = df['Churn']                 # the column we want the model to predict

# The split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% held out for testing
    random_state=42,    # makes the split reproducible — same result every run
    stratify=y          # keeps churn ratio equal in both halves
)

# Always verify the split worked as expected
print(f"Training rows:  {len(X_train)}")   # ~5625
print(f"Test rows:      {len(X_test)}")    # ~1407

# Verify stratification preserved the churn ratio
print("\nChurn rate in training set:", y_train.mean().round(3))
print("Churn rate in test set:    ", y_test.mean().round(3))
# Both should be ~0.265 — the same as the full dataset

# Compare Random Forest and Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold cross-validation — splits data 5 ways and averages the scores
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')

print(f"CV ROC-AUC scores: {scores.round(3)}")
print(f"Mean: {scores.mean():.3f}  ±  {scores.std():.3f}")

# ── Step 1: Scale features for logistic regression ────────────────────────────
# Logistic regression is sensitive to feature scale — a column ranging 0-8000
# (TotalCharges) will dominate one ranging 0-2 (SeniorCitizen) without scaling.
# Random forest doesn't care about scale, so we skip it there.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit AND transform on train
X_test_scaled  = scaler.transform(X_test)        # transform ONLY on test — never fit_transform

# ── Step 2: Train logistic regression ─────────────────────────────────────────
lr = LogisticRegression(
    max_iter=1000,      # default 100 often fails to converge on this dataset — bump it up
    random_state=42,
    C=1.0               # regularisation strength — 1.0 is the default, fine to start
)
lr.fit(X_train_scaled, y_train)
print("Logistic regression trained.")

# Quick cross-validation score on training data
lr_cv = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='roc_auc')
print(f"LR  CV ROC-AUC: {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")

# ── Step 3: Train random forest ───────────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=100,   # number of trees — 100 is a solid starting point
    max_depth=None,     # trees grow until leaves are pure — can overfit, but good baseline
    min_samples_leaf=1, # minimum samples required at a leaf node
    random_state=42,
    n_jobs=-1           # use all CPU cores — speeds up training significantly
)
rf.fit(X_train, y_train)   # no scaling needed
print("\nRandom forest trained.")

rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"RF  CV ROC-AUC: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

# ── Step 4: Compare both models side by side ──────────────────────────────────
from sklearn.metrics import roc_auc_score

results = pd.DataFrame({
    'Model':        ['Logistic Regression', 'Random Forest'],
    'CV ROC-AUC':   [lr_cv.mean(),          rf_cv.mean()],
    'CV Std':       [lr_cv.std(),            rf_cv.std()],
    'Test ROC-AUC': [
        roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]),
        roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    ]
}).round(3)

print("\n", results.to_string(index=False))

# Model Evaluation (ROC curve, confusion matrix, classification report)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)


# ── Step 1: Generate predictions from both models
# predict()       → hard class labels (0 or 1) — used for confusion matrix
# predict_proba() → probability scores       — used for ROC curve
# [:, 1] takes the probability of the POSITIVE class (churn = 1)

lr_preds  = lr.predict(X_test_scaled)
lr_probs  = lr.predict_proba(X_test_scaled)[:, 1]

rf_preds  = rf.predict(X_test)
rf_probs  = rf.predict_proba(X_test)[:, 1]

# ── Step 2: Classification report ────────────────────────────────────────────
# The most information-dense single output in sklearn evaluation.
# Shows precision, recall, and F1 for EACH class separately.

print("=" * 55)
print("LOGISTIC REGRESSION")
print("=" * 55)
print(classification_report(y_test, lr_preds,
                             target_names=['Stayed (0)', 'Churned (1)']))

print("=" * 55)
print("RANDOM FOREST")
print("=" * 55)
print(classification_report(y_test, rf_preds,
                             target_names=['Stayed (0)', 'Churned (1)']))

# ── Step 3: Understand what those numbers mean

# Use this block as a reference — not code we run, but concepts to know cold.

# For the CHURNED class (the one that matters for business action):
#
# Precision = of everyone we PREDICTED would churn, how many actually did?
#   → "Are we wasting retention budget on false alarms?"
#   → Precision 0.67 means 33% of our churn alerts are wrong
#
# Recall = of everyone who ACTUALLY churned, how many did we catch?
#   → "Are we missing churners who leave without intervention?"
#   → Recall 0.53 means we miss 47% of real churners
#
# F1 = harmonic mean of precision and recall — single balanced score
#   → Useful when classes are imbalanced and we care about both
#
# The business tradeoff: we usually can't maximise both.
# Raising the classification threshold → higher precision, lower recall
# Lowering the threshold             → lower precision, higher recall
#
# For churn: missing a churner (false negative) costs more than
# a wasted retention call (false positive) — so you'd lean toward recall.

# ── Step 4: Confusion matrices for both models

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, preds, title in zip(
    axes,
    [lr_preds, rf_preds],
    ['Logistic Regression', 'Random Forest']
):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Stayed', 'Churned']
    )
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{title}\nAccuracy: {(preds == y_test).mean():.3f}')

plt.suptitle('Confusion Matrices', fontsize=13, y=.95)
plt.tight_layout()
plt.show()

# ── Step 5: ROC curves — both models on one chart

fig, ax = plt.subplots(figsize=(7, 6))

for probs, label, color in [
    (lr_probs, 'Logistic Regression', 'steelblue'),
    (rf_probs, 'Random Forest',       'coral')
]:
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, label=f'{label}  (AUC = {auc:.3f})', color=color, lw=2)

# Random chance baseline
ax.plot([0, 1], [0, 1], 'k--', label='Random Chance (AUC = 0.500)', lw=1)

ax.set_xlabel('False Positive Rate\n(% of stayers incorrectly flagged as churners)')
ax.set_ylabel('True Positive Rate\n(% of churners correctly identified)')
ax.set_title('ROC Curve — Churn Prediction Models')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── Step 6: Find the optimal threshold
# Default sklearn threshold is 0.5 — but that's rarely optimal for imbalanced data.
# The point closest to (0, 1) on the ROC curve is the best balance of TPR and FPR.

fpr, tpr, thresholds = roc_curve(y_test, lr_probs)
optimal_idx = np.argmax(tpr - fpr)             # Youden's J statistic
optimal_threshold = thresholds[optimal_idx]

print(f"Default threshold (0.50):  ", end="")
print(classification_report(y_test, lr_preds,
      target_names=['Stayed','Churned'], output_dict=True)['Churned'])

# Apply optimal threshold
lr_preds_opt = (lr_probs >= optimal_threshold).astype(int)
print(f"\nOptimal threshold ({optimal_threshold:.2f}):")
print(classification_report(y_test, lr_preds_opt,
      target_names=['Stayed', 'Churned']))

# ── Step 7: Summary comparison table


from sklearn.metrics import precision_score, recall_score, f1_score

summary = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'ROC-AUC': [
        roc_auc_score(y_test, lr_probs),
        roc_auc_score(y_test, rf_probs)
    ],
    'Precision (churn)': [
        precision_score(y_test, lr_preds),
        precision_score(y_test, rf_preds)
    ],
    'Recall (churn)': [
        recall_score(y_test, lr_preds),
        recall_score(y_test, rf_preds)
    ],
    'F1 (churn)': [
        f1_score(y_test, lr_preds),
        f1_score(y_test, rf_preds)
    ]
})

print(summary.to_string(index=False))

# Final Summary

import matplotlib.patches as mpatches

# ── Part 1: Random Forest feature importance 
# Random forest gives importance as "mean decrease in impurity" —
# how much each feature reduces uncertainty across all trees

importances = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

top15 = importances.head(15).sort_values()  # sort ascending for horizontal bar

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['coral' if i >= len(top15) - 5 else 'steelblue'
          for i in range(len(top15))]   # highlight top 5 in coral

top15.plot(kind='barh', ax=ax, color=colors)

ax.set_title('Top 15 Features Driving Churn Prediction\n(Random Forest)',
             fontsize=13, pad=12)
ax.set_xlabel('Feature importance (mean decrease in impurity)')
ax.set_ylabel('')

# Add value labels on each bar
for i, (val, name) in enumerate(zip(top15.values, top15.index)):
    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9)

# Legend
blue_patch  = mpatches.Patch(color='steelblue', label='Contributing feature')
coral_patch = mpatches.Patch(color='coral',     label='Top 5 drivers')
ax.legend(handles=[blue_patch, coral_patch], loc='lower right')

plt.tight_layout()
plt.show()

# ── Part 2: Logistic regression coefficients 
# Coefficients show DIRECTION — positive means increases churn probability,
# negative means decreases it. RF importance has no direction.

coefs = pd.Series(
    lr.coef_[0],
    index=X_train.columns
).sort_values()

# Top 10 positive (churn drivers) and top 10 negative (churn reducers)
top_pos = coefs.tail(10)
top_neg = coefs.head(10)
plot_coefs = pd.concat([top_neg, top_pos]).sort_values()

colors = ['steelblue' if c < 0 else 'coral' for c in plot_coefs.values]

fig, ax = plt.subplots(figsize=(9, 7))
plot_coefs.plot(kind='barh', ax=ax, color=colors)

ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Logistic Regression Coefficients\n(after scaling — direction matters)',
             fontsize=13, pad=12)
ax.set_xlabel('Coefficient Value  (positive = increases churn probability)')

blue_patch  = mpatches.Patch(color='steelblue', label='Reduces churn risk')
coral_patch = mpatches.Patch(color='coral',     label='Increases churn risk')
ax.legend(handles=[blue_patch, coral_patch])

plt.tight_layout()
plt.show()

# ── Part 3: Segment analysis — what the features actually mean in the data ────
# This is what turns a model into a business recommendation.
# Don't just plot importances — show the actual churn rates by segment.

# Re-attach raw categorical columns for readability
df_orig = pd.read_csv('Datasets/Telco Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_orig['TotalCharges'] = pd.to_numeric(df_orig['TotalCharges'], errors='coerce')
df_orig.dropna(inplace=True)
df_orig['Churn_binary'] = (df_orig['Churn'] == 'Yes').astype(int)

# Tenure buckets — converts a continuous variable into readable segments
df_orig['Tenure_band'] = pd.cut(
    df_orig['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-12 mo', '13-24 mo', '25-48 mo', '49-72 mo']
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Churn Rate By Key Segments — Business Interpretation',
             fontsize=13, y=1.00)

# Panel 1: Contract type
churn_contract = df_orig.groupby('Contract')['Churn_binary'].mean().sort_values(ascending=False)
churn_contract.plot(kind='bar', ax=axes[0,0], color='coral')
axes[0,0].set_title('By Contract Type')
axes[0,0].set_ylabel('Churn Rate')
axes[0,0].set_ylim(0, 0.55)
axes[0,0].tick_params(axis='x', rotation=15)
for i, v in enumerate(churn_contract):
    axes[0,0].text(i, v + 0.01, f'{v:.0%}', ha='center', fontsize=10)

# Panel 2: Tenure band
churn_tenure = df_orig.groupby('Tenure_band', observed=True)['Churn_binary'].mean()
churn_tenure.plot(kind='bar', ax=axes[0,1], color='steelblue')
axes[0,1].set_title('By Tenure Band')
axes[0,1].set_ylabel('Churn Rate')
axes[0,1].set_ylim(0, 0.65)
axes[0,1].tick_params(axis='x', rotation=0)
for i, v in enumerate(churn_tenure):
    axes[0,1].text(i, v + 0.01, f'{v:.0%}', ha='center', fontsize=10)

# Panel 3: Internet service
churn_internet = df_orig.groupby('InternetService')['Churn_binary'].mean().sort_values(ascending=False)
churn_internet.plot(kind='bar', ax=axes[1,0], color='coral')
axes[1,0].set_title('By Internet Service Type')
axes[1,0].set_ylabel('Churn Rate')
axes[1,0].set_ylim(0, 0.55)
axes[1,0].tick_params(axis='x', rotation=0)
for i, v in enumerate(churn_internet):
    axes[1,0].text(i, v + 0.01, f'{v:.0%}', ha='center', fontsize=10)

# Panel 4: High-risk segment — month-to-month + fiber + under 12 months
high_risk = df_orig[
    (df_orig['Contract'] == 'Month-to-month') &
    (df_orig['InternetService'] == 'Fiber optic') &
    (df_orig['tenure'] <= 12)
]
low_risk = df_orig[
    (df_orig['Contract'] == 'Two year') &
    (df_orig['tenure'] > 24)
]

segment_rates = pd.Series({
    'High risk\n(MTM + Fiber\n< 12 mo)': high_risk['Churn_binary'].mean(),
    'All customers': df_orig['Churn_binary'].mean(),
    'Low risk\n(2yr contract\n> 24 mo)': low_risk['Churn_binary'].mean()
})
colors_seg = ['coral', 'steelblue', 'mediumseagreen']
segment_rates.plot(kind='bar', ax=axes[1,1], color=colors_seg)
axes[1,1].set_title('High vs Low Risk Segments')
axes[1,1].set_ylabel('Churn Rate')
axes[1,1].set_ylim(0, 0.85)
axes[1,1].tick_params(axis='x', rotation=0)
for i, v in enumerate(segment_rates):
    axes[1,1].text(i, v + 0.01, f'{v:.0%}', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# ── Part 4: Print the portfolio commentary ────────────────────────────────────
commentary = """
## Model interpretation & business recommendations

### What the model learned
Tenure, monthly charges, and contract type are the three dominant drivers of
churn, accounting for the majority of predictive signal in both models. This
aligns with intuition: customers who are new, paying a premium, and not locked
into a contract have the lowest switching cost and the highest motivation to leave.

### Key findings by segment
- Month-to-month customers churn at 43% vs 3% for two-year contracts — a 14x
  difference. Contract type is the single most actionable lever available.
- Customers in their first 12 months churn at 48%, dropping to under 15% by
  month 25. The first year is the highest-risk window by a significant margin.
- Fiber optic customers churn at 42% vs 19% for DSL users, suggesting either
  a pricing or service quality issue specific to that product line.
- The highest-risk segment — month-to-month, fiber optic, under 12 months —
  churns at approximately 70%. This group represents the clearest intervention
  target.

### Business recommendations
1. Early-tenure retention programme: trigger a proactive outreach sequence at
   months 3, 6, and 9 for month-to-month customers. The model can score every
   new customer at signup to prioritise outreach budget.

2. Contract conversion incentive: offering a discount or benefit in exchange for
   switching from month-to-month to a one-year contract would directly reduce
   the highest-churn segment. Even a 10% conversion rate on month-to-month
   customers would meaningfully reduce overall churn.

3. Fiber optic investigation: the elevated churn rate among fiber customers
   warrants a qualitative investigation — NPS data, support ticket analysis,
   or customer exit surveys — to determine whether this is a pricing, reliability,
   or expectation mismatch issue.

### Model limitations
- The model was trained on a single snapshot dataset with no temporal dimension.
  Churn propensity may vary by season, competitive environment, or price changes
  not captured here.
- TotalCharges and tenure are highly correlated. In production, tenure alone
  may be sufficient and reduces collinearity.
- The 0.5 classification threshold was adjusted to 0.38 to increase recall on
  the minority class, accepting more false positives in exchange for catching
  more true churners — appropriate given that the cost of missing a churner
  exceeds the cost of a wasted retention call.
"""

print(commentary)