# ============================================================
# CREDIT CARD FRAUD DETECTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
df = pd.read_csv(r'E:\ARCH TECH Internship\Task 4\creditcard.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\nFraud Percentage: {df['Class'].mean()*100:.4f}%")

# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Plot class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette=['steelblue','red'])
plt.title('Class Distribution (0=Normal, 1=Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Transaction amount by class
plt.figure(figsize=(8,4))
df[df['Class']==0]['Amount'].hist(bins=50, alpha=0.6, label='Normal', color='blue')
df[df['Class']==1]['Amount'].hist(bins=50, alpha=0.6, label='Fraud', color='red')
plt.legend()
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.savefig('amount_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlation heatmap (sample)
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr[['Class']].sort_values('Class', ascending=False),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation with Class')
plt.savefig('correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# STEP 3: PREPROCESSING
# ============================================================

# Scale 'Amount' and 'Time' (V1-V28 are already scaled by PCA)
scaler = StandardScaler()
df['scaled_Amount'] = scaler.fit_transform(df[['Amount']])
df['scaled_Time']   = scaler.fit_transform(df[['Time']])

# Drop original columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# Features & Target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining size: {X_train.shape}")
print(f"Testing size:  {X_test.shape}")

# ============================================================
# STEP 4: HANDLE CLASS IMBALANCE USING SMOTE
# ============================================================

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Normal: {sum(y_train_res==0)}, Fraud: {sum(y_train_res==1)}")

# ============================================================
# STEP 5: TRAIN MODELS
# ============================================================

# --- Model 1: Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

# --- Model 2: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

# ============================================================
# STEP 6: EVALUATE MODELS
# ============================================================

def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=['Normal','Fraud']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Fraud'],
                yticklabels=['Normal','Fraud'])
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'confusion_matrix_{name.replace(" ","_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
evaluate_model("Random Forest",       y_test, y_pred_rf, y_prob_rf)

# ============================================================
# STEP 7: ROC CURVE COMPARISON
# ============================================================

plt.figure(figsize=(8,6))
for name, prob in [("Logistic Regression", y_prob_lr), ("Random Forest", y_prob_rf)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

plt.plot([0,1],[0,1],'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# STEP 8: FEATURE IMPORTANCE (Random Forest)
# ============================================================

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(15)

plt.figure(figsize=(8,5))
feat_imp.plot(kind='bar', color='steelblue')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ All done! Check saved plots for your report.")