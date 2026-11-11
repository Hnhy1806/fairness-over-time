import pandas as pd
import numpy as np
from src.fairness import make_sensitive
from scipy.stats import ks_2samp
from src.preprocess import FEATURE_COLS, NUMERIC_COLS

# CPI-U annual averages, BLS series CUUR0000SA0
CPI_U = {2014: 236.736, 2015: 237.017, 2016: 240.007, 2017: 245.120}
KS_COLS = ["AGEP", "WKHP", "SCHL"]

def label_rates(income, year, cpi=CPI_U, base_year=2014, threshold=50_000):
    indexed_threshold = threshold * cpi[year] / cpi[base_year]
    return {
        "year": year,
        "fixed_rate": (income > threshold).mean(),
        "indexed_rate": (income > indexed_threshold).mean(),
    }

def label_rates_by_group(meta, year, cpi=CPI_U, base_year=2014, threshold=50_000):
    """Fixed and CPI-indexed positive rates: overall, by sex, by race_eth.

    meta: output of load_acs for one year (uses income, SEX, race_eth).
    Returns long format for label_drift.csv: year, attribute, group, metric, value.
    Overall rows have attribute = "overall", group = "all".
    """
    income = meta["income"]
    sens = make_sensitive(meta)

    groupings = {
        "overall": pd.Series("all", index=meta.index),
        "sex": sens["sex"],
        "race_eth": sens["race_eth"],
    }

    rows = []
    for attribute, labels in groupings.items():
        for group in sorted(labels.unique()):
            r = label_rates(income[labels == group], year, cpi, base_year, threshold)
            for metric in ["fixed_rate", "indexed_rate"]:
                rows.append({
                    "year": year, "attribute": attribute, "group": group,
                    "metric": metric, "value": r[metric],
                })
    return pd.DataFrame(rows)

MIN_CATEGORY_COUNT = 30   # same rule as the preprocessor's min_frequency
EPS = 1e-4                # replaces empty-bin shares so ln(0) can't happen


def make_bins(base_col, numeric):
    """Bins fitted on the 2014 training data only.

    Numeric: decile edges (duplicates removed, open ends).
    Categorical: categories with at least MIN_CATEGORY_COUNT rows; the rest become "other".
    """
    if numeric:
        edges = np.quantile(base_col, np.linspace(0.1, 0.9, 9))
        return np.unique(np.concatenate([[-np.inf], edges, [np.inf]]))
    counts = base_col.value_counts()
    return counts[counts >= MIN_CATEGORY_COUNT].index


def assign_bins(col, bins, numeric):
    """Put any year's values into the 2014 bins. Returns bin labels as strings."""
    if numeric:
        return pd.cut(col, bins).astype(str)
    return col.where(col.isin(bins), "other").astype(str)


def psi(base_binned, cur_binned, eps=EPS):
    """Population Stability Index between two binned columns (0 = identical)."""
    all_bins = sorted(set(base_binned) | set(cur_binned))
    p = base_binned.value_counts(normalize=True).reindex(all_bins, fill_value=0).clip(lower=eps)
    q = cur_binned.value_counts(normalize=True).reindex(all_bins, fill_value=0).clip(lower=eps)
    return float(((q - p) * np.log(q / p)).sum())

def feature_drift(X_base, X_cur, year):
    """PSI for every feature and KS for ordered features, X_cur vs. the 2014 training data.

    Bins are fitted on X_base only. Returns the drift_over_time.csv format:
    year, feature, metric, value.
    """
    rows = []
    for feature in FEATURE_COLS:
        numeric = feature in NUMERIC_COLS
        bins = make_bins(X_base[feature], numeric)
        value = psi(assign_bins(X_base[feature], bins, numeric),
                    assign_bins(X_cur[feature], bins, numeric))
        rows.append({"year": year, "feature": feature, "metric": "PSI", "value": value})

        if feature in KS_COLS:
            d = ks_2samp(X_base[feature], X_cur[feature]).statistic
            rows.append({"year": year, "feature": feature, "metric": "KS", "value": float(d)})

    return pd.DataFrame(rows)