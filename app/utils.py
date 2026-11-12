"""Cached loaders: the only place the app reads artifacts.

The app never trains. Every path comes from src.data, and the models come from
src.models.load_models(), so the app uses exactly what the notebooks saved.
"""
import pandas as pd
import streamlit as st

from src.data import (
    METRICS_PATH,
    CI_PATH,
    LABEL_DRIFT_PATH,
    DRIFT_PATH,
    SHAP_PATH,
    FIX_PATH,
    BACKGROUND_PATH,
)
from src.models import load_models

# Results files the app reads (name -> path), used by pages and the artifact check
RESULT_FILES = {
    "metrics_over_time.csv": METRICS_PATH,
    "ci_over_time.csv": CI_PATH,
    "label_drift.csv": LABEL_DRIFT_PATH,
    "drift_over_time.csv": DRIFT_PATH,
    "shap_over_time.csv": SHAP_PATH,
    "fix_comparison.csv": FIX_PATH,
}


@st.cache_data
def load_results(path):
    """Read one results CSV. Cached per path; each caller gets its own copy."""
    return pd.read_csv(path)


@st.cache_resource
def get_models():
    """Load the 4 frozen 2014 models once, shared across reruns and users.

    Returns {name: Pipeline} with names lr, lr_mitigated, xgb, xgb_mitigated.
    """
    return load_models()


@st.cache_data
def load_background():
    """500 raw 2014 training rows used as the SHAP background (Try a Profile)."""
    return pd.read_csv(BACKGROUND_PATH)