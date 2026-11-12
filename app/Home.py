import pandas as pd
import streamlit as st

from utils import RESULT_FILES, get_models, load_background, load_results

st.set_page_config(page_title="Fairness Over Time", layout="wide")

st.title("Fairness Over Time")
st.markdown(
    "**Does a model that is fair when deployed stay fair as data changes over time?**  \n"
    "Income models (> $50K) trained on Texas ACS data from 2014, "
    "monitored on 2015–2017 without retraining."
)
st.info("Work in progress: page content is being built.")

# Temporary smoke test (remove at A7)
with st.expander("Artifact check (temporary)"):
    rows = [{"file": name, "rows": len(load_results(path))} for name, path in RESULT_FILES.items()]
    rows.append({"file": "background_sample.csv", "rows": len(load_background())})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.write("Models loaded:", list(get_models()))