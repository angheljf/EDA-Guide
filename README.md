# Exploratory Data Analysis (EDA) Guide in Python

This guide provides a step-by-step outline for performing **Exploratory Data Analysis (EDA)** in Python. EDA is a critical step in understanding your data, identifying patterns, and preparing it for modeling.

---

## Table of Contents
1. [Understand the Problem and Data](#1-understand-the-problem-and-data)
2. [Handle Missing Data](#2-handle-missing-data)
3. [Data Cleaning](#3-data-cleaning)
4. [Univariate Analysis](#4-univariate-analysis)
5. [Bivariate and Multivariate Analysis](#5-bivariate-and-multivariate-analysis)
6. [Feature Engineering](#6-feature-engineering)
7. [Visualize Insights](#7-visualize-insights)
8. [Document Findings](#8-document-findings)

---

## 1. Understand the Problem and Data

### Define the Objective
- Clearly articulate the goals of your analysis. Are you trying to predict a target variable, identify trends, or uncover patterns?
- Understand the context of the data (e.g., business problem, scientific question).

### Gather Data
Load the dataset into Python using `pandas`:
```python
import pandas as pd
df = pd.read_csv('data.csv')
```
**For other formats:**
- **Excel:** `pd.read_excel('data.xlsx')`
- **SQL:** `pd.read_sql('SELECT * FROM data', connection)`

### Inspect the Data
- **First few rows:** `df.head()`
- **Data types:** `df.info()`
- **Summary statistics:** `df.describe()`
- **Shape of the data:** `df.shape`
- **Unique values:** `df['column'].unique()` or `df['column'].value_counts()`

## 2. Handle Missing Data

### Identify Missing Data
```python
df.isnull().sum()   # Count of missing values
df.isnull().mean()  # Percentage of missing values
```
### Decide on a Strategy
- **Drop missing values:** `df.dropna(inplace=True)`
- **Fill missing values:**
    - Fill with a constant: `df.fillna(0, inplace=True)`
    - Fill with mean/median: `df['column'].fillna(df['column'].mean(), inplace=True)`
    - Forward-fill/back-fill: `df.fillna(method='ffill', inplace=True)`
- **Impute missing values:** 
```python
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

## 3. Data Cleaning

### Remove Duplicates
```python
df.duplicated().sum()  # Count of duplicates
df.drop_duplicates(inplace=True) # Remove duplicates
```
### Handle Outliers
- **Visualize outliers:** Box plots, histograms, scatter plots
```python
import seaborn as sns
sns.boxplot(x=df['column'])
```
- **Remove outliers:** 
```python
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['column'] >= Q1-1.5*IQR) & (df['column'] <= Q3+1.5*IQR)]
```
### Correct Data Types
```python
df['column'] = df['column'].astype('category')  # Convert to categorical
df['date_column'] = pd.to_datetime(df['date_column'])  # Convert to datetime
```
### Standardize Column Names
```python
df.columns = df.columns.str.lower().str.replace(' ', '_')   # Lowercase and replace spaces
```

## 4. Univariate Analysis

### Analyze Individual Variables
- **Numerical Variables:**
```python
sns.histplot(df['numerical_column'], kde=True)  # Histogram
sns.boxplot(x=df['numerical_column'])  # Box plot
df['column'].describe()  # Summary statistics
```
- **Categorical Variables:**
```python
df['categorical_column'].value_counts().plot(kind='bar')  # Bar plot
df['categorical_column'].value_counts().plot(kind='pie')  # Pie chart
```

## 5. Bivariate and Multivariate Analysis

### Explore Relationships Between Variables
- **Numerical vs. Numerical:**
```python
sns.scatterplot(x='column1', y='column2', data=df)  # Scatter plot
df.corr()  # Correlation matrix
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')  # Correlation heatmap
```
- **Numerical vs. Categorical:**
```python
sns.boxplot(x='categorical_column', y='numerical_column', data=df)  # Box plot
sns.violinplot(x='categorical_column', y='numerical_column', data=df)  # Violin plot
```
- **Categorical vs. Categorical:**
```python
pd.crosstab(df['cat_column1'], df['cat_column2'])   # Cross-tabulation
sns.countplot(x='cat_column1', hue='cat_column2', data=df)  # Stacked bar plot
sns.heatmap(pd.crosstab(df['cat_column1'], df['cat_column2']), cmap='coolwarm', annot=True)  # Heatmap
```

## 6. Feature Engineering

### Create New Features
```python
df['new_feature'] = df['feature1'] + df['feature2']  # Combine features
df['log_feature'] = np.log(df['feature'])  # Log transformation
df['age'] = 2022 - df['birth_year']  # Calculate age
```
### Transform Variables
- **Normalize or Scale:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['scaled_column'] = scaler.fit_transform(df[['numerical_column']])
```
- **Log Transformation:**
```python
df['log_column'] = np.log(df['numerical_column'])
```

## 7. Visualize Insights

Use visualizations to communicate findings effectively:
- **Line Plot:** Trends over time
- **Bar Plot:** Comparing categories
- **Histogram:** Distribution of numerical variables
- **Box Plot:** Distribution and outliers
- **Heatmap:** Correlation between variables
- **Scatter Plot:** Relationship between variables
- **Pair Plot:** Pairwise relationships in multivariate data

## 8. Document Findings

- Summarize key insights and trends:
    - Trends, patterns and anomalies
    - Relationships between variables
- Note assumptions and limitations of the analysis:
    - Data quality issues (missing values, outliers)
    - Assumptions made during analysis
    - Potential biases or errors
- Prepare visualizations and tables to support your findings:
    - Include titles, axis labels, legends
    - Highlight key points with annotations
    - Use consistent color schemes and styles
