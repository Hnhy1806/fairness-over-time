import pandas as pd
import numpy as np
import shap

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from src.preprocess import FEATURE_COLS

def explain(pipe, X, background):
    #split the pipeline
    pre = pipe.named_steps['pre']
    model = pipe.named_steps['model']

    #transform the data
    Xt = pre.transform(X)
    background_t = pre.transform(background)
    feature_names = pre.get_feature_names_out()

    #shap for each model type
    if isinstance(model, XGBClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xt)
    elif isinstance(model, LogisticRegression):
        masker = shap.maskers.Independent(background_t, max_samples=len(background_t))
        explainer = shap.LinearExplainer(model, masker)
        shap_values = explainer.shap_values(Xt)
    else:
        raise ValueError('Model type not supported')
    
    #convert SHAP values to DataFrame
    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names,
        index=X.index
    )

    #map encoded features back to original features
    mapping = {}

    for name in feature_names:
        for col in FEATURE_COLS:
            if name == col or name.startswith(col + "_"):
                mapping[name] = col
                break
        else:
            raise ValueError(f"Unknown transformed feature: {name}")

    #combine SHAP values belonging to the same original feature
    grouped_shap = (
        shap_df.T
        .groupby(mapping)
        .sum()
        .T
    )

    grouped_shap = grouped_shap[FEATURE_COLS]

    # Step 5: Preserve original row index
    grouped_shap.index = X.index

    return grouped_shap 
    
def shap_importance(shap_df, year, model):
    """Mean |SHAP| and mean SHAP per raw feature, in shap_over_time.csv format.

    shap_df: output of explain() (rows = people, columns = the 10 raw features).
    Returns: year, model, feature, metric, value.
    """
    summary = pd.DataFrame({
        "mean_abs_shap": shap_df.abs().mean(),   # how much the model relies on the feature
        "mean_shap": shap_df.mean(),             # net direction: pushes scores up (+) or down (-)
    })
    summary.index.name = "feature"

    rows = summary.reset_index().melt(id_vars="feature", var_name="metric", value_name="value")
    rows["value"] = rows["value"].astype(float)  # XGBoost returns float32
    rows.insert(0, "model", model)
    rows.insert(0, "year", year)
    return rows