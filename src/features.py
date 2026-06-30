"""
features.py — Feature engineering utilities for khipu cord and knot data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Label-encode a list of categorical columns in place."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def aggregate_cord_features(cords_df: pd.DataFrame, group_col: str = "khipu_id") -> pd.DataFrame:
    """
    Aggregate cord-level features to khipu level.

    Parameters
    ----------
    cords_df : DataFrame with one row per cord.
    group_col : column identifying each khipu.

    Returns
    -------
    DataFrame with one row per khipu and summary statistics.
    """
    numeric_cols = cords_df.select_dtypes(include=np.number).columns.tolist()
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)

    agg = cords_df.groupby(group_col)[numeric_cols].agg(["mean", "std", "min", "max", "sum"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    return agg


def drop_low_variance(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Remove columns with variance at or below *threshold*."""
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    numeric = df.select_dtypes(include=np.number)
    selector.fit(numeric)
    kept = numeric.columns[selector.get_support()].tolist()
    non_numeric = df.select_dtypes(exclude=np.number).columns.tolist()
    return df[non_numeric + kept]
