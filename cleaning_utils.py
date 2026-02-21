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
