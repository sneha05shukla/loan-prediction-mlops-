# 1. Import Packages
import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# 2. Read Data
credit_df = pd.read_csv("./data/credit_train.csv")

# 3. Data Processing & Cleansing

# Fill missing values
credit_df['Months since last delinquent'] = credit_df['Months since last delinquent'].fillna(0)

# Correct Credit Score values > 900
credit_df.loc[credit_df['Credit Score'] >= 900, 'Credit Score'] = (
    credit_df.loc[credit_df['Credit Score'] >= 900, 'Credit Score'] / 10
)

# Fill missing Credit Score
credit_df['Credit Score'] = credit_df['Credit Score'].fillna(
    round(credit_df['Credit Score'].mean())
)

# Fill missing Annual Income
credit_df['Annual Income'] = credit_df['Annual Income'].fillna(
    round(credit_df['Annual Income'].mean())
)

# Clean Years in current job
credit_df['Years in current job'] = (
    credit_df['Years in current job']
    .str.replace(' years', '', regex=False)
    .str.replace(' year', '', regex=False)
    .str.replace('< 1', '0.5', regex=False)
    .str.replace('+', '', regex=False)
)

credit_df['Years in current job'] = credit_df['Years in current job'].astype(float)

credit_df['Years in current job'].fillna(
    credit_df['Years in current job'].median(),
    inplace=True
)

# Remove remaining nulls
credit_df.dropna(inplace=True)

print("\nDataset Head:")
print(credit_df.head())

print("\nDataset Statistics:")
print(credit_df.describe())

# 4. Drop Unwanted Columns
credit_df.drop(['Loan ID', 'Customer ID'], axis=1, inplace=True)

# 5. Encode Categorical Columns

categorical_cols = credit_df.select_dtypes(include=['object']).columns

print("\nCategorical Columns:")
print(list(categorical_cols))

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    credit_df[col] = le.fit_transform(credit_df[col].astype(str))
    label_encoders[col] = le

# 6. Features and Target
X = credit_df.drop('Loan Status', axis=1)
Y = credit_df['Loan Status']

print("\nFeature Data Types:")
print(X.dtypes)

print("\nTarget Data Type:")
print(Y.dtype)

# 7. Apply SMOTE
smote = SMOTE(random_state=42)

transformed_feature, transformed_label = smote.fit_resample(X, Y)

print("\nAfter SMOTE:")
print("Features Shape :", transformed_feature.shape)
print("Labels Shape   :", transformed_label.shape)

# 8. Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    transformed_feature,
    transformed_label,
    test_size=0.2,
    random_state=2
)

# 9. Train Model
clf_tree_best = DecisionTreeClassifier(
    ccp_alpha=0.001,
    criterion='gini',
    max_depth=20,
    random_state=2
)

clf_tree_best.fit(X_train, Y_train)

# 10. Predictions
Y_pred = clf_tree_best.predict(X_test)

print("\nAccuracy Score:")
print(accuracy_score(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

# 11. Save Model
joblib.dump(clf_tree_best, "./model/loan_default.pkl")

print("\nModel Saved Successfully!")


# save test sample as csv for prediction
X_test.to_csv("./data/x_test_sample.csv")

