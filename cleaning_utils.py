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
