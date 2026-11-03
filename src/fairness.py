"""Fairness evaluation: works on any data + predictions (any year, any model)."""
import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame, count, selection_rate, true_positive_rate, false_positive_rate,
)
from sklearn.metrics import accuracy_score

METRICS = {
    'Count': count,
    'Selection_rate': selection_rate,
    'TPR': true_positive_rate,
    'FPR': false_positive_rate,
    'Accuracy': accuracy_score
}


def group_metrics(y_true, y_pred, sensitive):
    mf = MetricFrame(
        metrics=METRICS,             
        y_true=y_true,
        y_pred=y_pred,              
        sensitive_features=sensitive, 
    )
    return mf.by_group

def fairness_summary(y_true, y_pred, sensitive):
    mf = MetricFrame(
        metrics={
            'Selection_rate': selection_rate,
            'TPR': true_positive_rate,
            'FPR': false_positive_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    gaps = mf.difference()
    return {
        'DPD': gaps['Selection_rate'],
        'EOD': max(gaps['TPR'], gaps['FPR'])
    }

SEX_LABELS = {1: "Male", 2: "Female"}

def make_sensitive(meta):
    """Sensitive attributes with readable labels, for fairness evaluation."""
    return pd.DataFrame({
        "sex": meta["SEX"].map(SEX_LABELS),
        "race_eth": meta["race_eth"],
    })

ATTRIBUTES = {
    "sex": ["sex"],
    "race_eth": ["race_eth"],
    "race_eth_x_sex": ["race_eth", "sex"],
}

def _group_label(g):
    """('Hispanic', 'Female') -> 'Hispanic × Female'; 'Female' -> 'Female'."""
    if isinstance(g, tuple):
        return " × ".join(str(x) for x in g)
    return str(g)

def fairness_long(y_true, y_pred, sensitive):
    """All per-group metrics and summary gaps in long format.

    Columns: attribute, group, metric, value.
    Summary gaps (dpd, eod) appear with group = "(gap)".
    sensitive: output of make_sensitive(), aligned with y_true and y_pred.
    """
    frames = []
    for attribute, cols in ATTRIBUTES.items():
        sens = sensitive[cols[0]] if len(cols) == 1 else sensitive[cols]

        # Per-group metrics: wide (one column per metric) -> long
        bg = group_metrics(y_true, y_pred, sens)
        bg.index = bg.index.map(_group_label)
        bg.index.name = "group"
        per_group = bg.reset_index().melt(
            id_vars="group", var_name="metric", value_name="value"
        )

        # Summary gaps as extra rows
        gaps = fairness_summary(y_true, y_pred, sens)
        gap_rows = pd.DataFrame({
            "group": "(gap)",
            "metric": list(gaps.keys()),
            "value": list(gaps.values()),
        })

        rows = pd.concat([per_group, gap_rows], ignore_index=True)
        rows.insert(0, "attribute", attribute)
        frames.append(rows)

    return pd.concat(frames, ignore_index=True)

#bootstrap confidence intervals
def bootstrap_gaps(y_true, y_pred, sensitive, n_boot=200, seed=42):
    """95% bootstrap confidence intervals for dpd and eod, per attribute.

    Resamples rows with replacement n_boot times and recomputes the gaps.
    Returns a DataFrame: attribute, metric, value, ci_low, ci_high.
    """
    # Use plain positions 0..n-1 everywhere so idx picks the same people from all inputs
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = sensitive.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    rows = []
    for attribute, cols in ATTRIBUTES.items():
        sens = sensitive[cols[0]] if len(cols) == 1 else sensitive[cols]
        point = fairness_summary(y_true, y_pred, sens)

        samples = {"DPD": [], "EOD": []}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)   # n random row positions, with replacement
            s = fairness_summary(
                y_true[idx],
                y_pred[idx],
                sens.iloc[idx].reset_index(drop=True),
            )
            samples["DPD"].append(s["DPD"])
            samples["EOD"].append(s["EOD"])

        for metric in ["DPD", "EOD"]:
            rows.append({
                "attribute": attribute,
                "metric": metric,
                "value": point[metric],
                "ci_low": np.nanpercentile(samples[metric], 2.5),
                "ci_high": np.nanpercentile(samples[metric], 97.5),
            })

    return pd.DataFrame(rows)