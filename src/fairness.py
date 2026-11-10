"""Fairness evaluation: works on any data + predictions (any year, any model)."""
import pandas as pd
import numpy as np
from fairlearn.metrics import (
    MetricFrame, count, selection_rate, true_positive_rate, false_positive_rate,
)
from sklearn.metrics import accuracy_score, roc_auc_score

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

#evaluate
def evaluate_model(pipe, X, y_true, meta):
    """Overall accuracy/AUC plus all fairness metrics for one fitted pipeline.

    Returns long format: attribute, group, metric, value.
    Overall rows have attribute = "overall", group = "all".
    """
    y_true = np.asarray(y_true)
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]

    overall = pd.DataFrame({
        "attribute": "overall",
        "group": "all",
        "metric": ["accuracy", "auc"],
        "value": [accuracy_score(y_true, pred), roc_auc_score(y_true, proba)],
    })
    fair = fairness_long(y_true, pred, make_sensitive(meta).reset_index(drop=True))
    return pd.concat([overall, fair], ignore_index=True)


def evaluate_all(models, X, y_true, meta, year, model_info):
    """Evaluate every model on one dataset and label the rows.

    Returns: year, model, mitigated, attribute, group, metric, value
    (the metrics_over_time.csv format). Works for any year.
    """
    frames = []
    for name, pipe in models.items():
        rows = evaluate_model(pipe, X, y_true, meta)
        rows.insert(0, "mitigated", model_info[name]["mitigated"])
        rows.insert(0, "model", model_info[name]["model"])
        rows.insert(0, "year", year)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)

#bootstrap
def bootstrap_all(models, X, y_true, meta, year, model_info, n_boot=200, seed=42):
    """Bootstrap CIs for DPD/EOD for every model on one dataset.

    Returns: year, model, mitigated, attribute, metric, value, ci_low, ci_high.
    """
    sensitive = make_sensitive(meta)
    y_true = np.asarray(y_true)
    frames = []
    for name, pipe in models.items():
        pred = pipe.predict(X)
        ci = bootstrap_gaps(y_true, pred, sensitive, n_boot=n_boot, seed=seed)
        ci.insert(0, "mitigated", model_info[name]["mitigated"])
        ci.insert(0, "model", model_info[name]["model"])
        ci.insert(0, "year", year)
        frames.append(ci)
    return pd.concat(frames, ignore_index=True)

#save the result
from pathlib import Path


def save_results(new: pd.DataFrame, path, keys=("year",)) -> pd.DataFrame:
    """Write results to a CSV, replacing any existing rows with the same keys.

    Rerunning the same year replaces that year's rows instead of adding
    duplicates. With keys=("year", "model", "mitigated"), only the models in
    `new` are replaced.
    """
    path = Path(path)
    keys = list(keys)

    if path.exists():
        old = pd.read_csv(path)

        if set(old.columns) != set(new.columns):
            raise ValueError(
                f"Column mismatch with {path.name}: "
                f"file has {sorted(old.columns)}, new has {sorted(new.columns)}"
            )

        # Rows in `old` whose key combination appears in `new` get removed
        new_keys = new[keys].drop_duplicates()
        merged = old.merge(new_keys, on=keys, how="left", indicator=True)
        kept = old[(merged["_merge"] == "left_only").to_numpy()]

        out = pd.concat([kept, new[old.columns]], ignore_index=True)
    else:
        out = new.copy()

    sort_cols = [c for c in ["year", "model", "mitigated", "attribute", "group", "metric"]
             if c in out.columns]
    
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out

