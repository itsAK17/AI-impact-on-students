
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


plt.style.use('default')
sns.set_theme()



df = pd.read_csv(r"C:\Users\amarj\Downloads\ai_student_impact_dataset (1).csv")


print("\nShape of Dataset:")
print(df.shape)



print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())



print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())



print("\nMissing Values:")
print(df.isnull().sum())

plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()



duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)



numerical_cols = df.select_dtypes(
    include=np.number
).columns

print("\nNumerical Features:")
print(list(numerical_cols))



categorical_cols = df.select_dtypes(
    include='object'
).columns

print("\nCategorical Features:")
print(list(categorical_cols))



for col in numerical_cols:

    plt.figure(figsize=(8,4))

    sns.histplot(
        df[col],
        bins=30,
        kde=True
    )

    plt.title(f"Distribution of {col}")
    plt.show()



for col in categorical_cols:

    plt.figure(figsize=(8,4))

    sns.countplot(
        x=df[col]
    )

    plt.title(f"Count Plot of {col}")

    plt.xticks(rotation=45)

    plt.show()



if ('Pre_GPA' in df.columns and
    'Post_GPA' in df.columns):

    df['GPA_Improvement'] = (
        df['Post_GPA']
        - df['Pre_GPA']
    )

    print("\nAverage GPA Improvement:")
    print(df['GPA_Improvement'].mean())

    plt.figure(figsize=(8,5))

    sns.histplot(
        df['GPA_Improvement'],
        bins=30,
        kde=True
    )

    plt.title("GPA Improvement Distribution")

    plt.show()



if ('Weekly_GenAI_Hours' in df.columns and
    'Post_GPA' in df.columns):

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        x='Weekly_GenAI_Hours',
        y='Post_GPA',
        data=df
    )

    plt.title(
        "Weekly GenAI Hours vs Post GPA"
    )

    plt.show()



if ('Prompt_Skill_Level' in df.columns and
    'Post_GPA' in df.columns):

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x='Prompt_Skill_Level',
        y='Post_GPA',
        data=df
    )

    plt.title(
        "Prompt Skill Level vs GPA"
    )

    plt.show()



if ('AI_Dependency_Level' in df.columns and
    'Skill_Retention_Score' in df.columns):

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x='AI_Dependency_Level',
        y='Skill_Retention_Score',
        data=df
    )

    plt.title(
        "AI Dependency vs Skill Retention"
    )

    plt.show()



if ('Burnout_Risk' in df.columns and
    'Weekly_GenAI_Hours' in df.columns):

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x='Burnout_Risk',
        y='Weekly_GenAI_Hours',
        data=df
    )

    plt.title(
        "Burnout Risk vs AI Usage"
    )

    plt.show()



df_corr = df.copy()

for col in df_corr.columns:

    if df_corr[col].dtype == 'object':

        df_corr[col] = pd.factorize(
            df_corr[col]
        )[0]

corr_matrix = df_corr.corr()

plt.figure(figsize=(14,10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Correlation Heatmap")

plt.show()




if 'GPA_Improvement' in df_corr.columns:

    print("\nCorrelation with GPA Improvement")

    correlations = (
        corr_matrix['GPA_Improvement']
        .sort_values(ascending=False)
    )

    print(correlations)



important_features = []

for col in [
    'Weekly_GenAI_Hours',
    'Traditional_Study_Hours',
    'Pre_GPA',
    'Post_GPA'
]:

    if col in df.columns:

        important_features.append(col)

if len(important_features) > 1:

    sns.pairplot(
        df[important_features]
    )

    plt.show()



df['GPA_Improvement'] = (
    df['Post_Semester_GPA']
    - df['Pre_Semester_GPA']
)

df['Improved'] = np.where(
    df['GPA_Improvement'] > 0,
    1,
    0
)



df_ml = df.copy()

for col in df_ml.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))



X = df_ml.drop([
    'Improved',
    'GPA_Improvement',
    'Pre_Semester_GPA',
    'Post_Semester_GPA'
], axis=1)

y = df_ml['Improved']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=50,
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)



y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]



print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print("Accuracy :", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred)*100, 2), "%")
print("Recall   :", round(recall_score(y_test, y_pred)*100, 2), "%")
print("F1 Score :", round(f1_score(y_test, y_pred)*100, 2), "%")
print("ROC AUC  :", round(roc_auc_score(y_test, y_prob)*100, 2), "%")



print("\nClassification Report")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()



importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Important Features")
print(importance.head(10))

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance.head(10),
    x='Importance',
    y='Feature'
)

plt.title("Top 10 Feature Importance")
plt.show()

print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

print("""
1. Examined the impact of AI usage on students.

2. Studied relationships among:
   - GPA
   - AI Usage
   - Study Hours
   - Prompt Skills
   - Burnout Risk

3. Identified factors contributing
   to academic improvement.

4. Visualized patterns using:
   - Histograms
   - Boxplots
   - Scatterplots
   - Heatmaps

5. Generated insights to support
   educational decision-making.
""")

print("\nPROJECT COMPLETED SUCCESSFULLY!")





