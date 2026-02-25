import pandas as pd
from typing import List

def get_null_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a professional summary of missing data.
    Essential for initial data auditing in Google-scale projects.
    """
    null_count = df.isnull().sum()
    null_percentage = (df.isnull().sum() / len(df)) * 100
    report = pd.concat([null_count, null_percentage], axis=1, keys=['Total Nulls', 'Percentage'])
    return report[report['Total Nulls'] > 0].sort_values(by='Percentage', ascending=False)
def get_categorical_counts(df, column_name):
    """
    Data Scientist Audit Tool: 
    Returns the frequency of each unique category in a column.
    Useful for checking data distribution before modeling.
    """
    return df[column_name].value_counts()
def get_correlation_matrix(df):
    """
    Advanced DS Tool: Computes the correlation matrix.
    Essential for Feature Selection in Machine Learning.
    """
    return df.corr()
import re

def clean_text(text):
    """
    Standardizes text by removing special characters 
    and converting to lowercase. Essential for NLP.
    """
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).lower().strip()

import time

def timer_decorator(func):
    """
    Pro Tool: Measures execution time of a function.
    Critical for identifying performance bottlenecks.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f}s")
        return result
    return wrapper

def standard_scaler(data):
    """
    Advanced DS Tool: Performs Z-score standardization.
    Essential for scaling features before training models like SVM or Logistic Regression.
    """
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return [(x - mean) / std_dev for x in data]
