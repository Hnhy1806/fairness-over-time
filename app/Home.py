import pandas as pd
import streamlit as st

from utils import RESULT_FILES, get_models, load_background, load_results

st.set_page_config(page_title="Fairness Over Time", layout="wide")

st.title("Fairness Over Time")
st.markdown(
    "### Does a model that is fair when deployed stay fair as data changes over time?\n"
    "Income models (does a person earn **more than $50K**?) are trained on **Texas** census data from "
    "**2014**, made fair with a fairness constraint, then **frozen** and run on 2015–2017 without "
    "retraining. This app follows what happens to their fairness, why it changes, and how to fix it."
)

# ---------------------------------------------------------------- research questions
st.subheader("Three questions")
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown(
            "**RQ1 · Does fairness hold over time?**\n\n"
            "Mostly, but it **erodes gradually and unevenly**. No model collapses, and mitigation's "
            "benefit survives: in 2017 the mitigated models' gaps are still far below the unmitigated ones."
        )
        st.page_link("pages/2_Fairness_Over_Time.py", label="Fairness Over Time", icon="📈")
with c2:
    with st.container(border=True):
        st.markdown(
            "**RQ2 · Why does it change?**\n\n"
            "**Concept drift meets a fixed cutoff.** Inputs barely change, but more people with the same "
            "features now earn over $50K (84% real income growth, 16% inflation). The 2014 cutoff misses "
            "them unevenly across groups."
        )
        st.page_link("pages/3_Why_It_Changed.py", label="Why It Changed", icon="🔍")
with c3:
    with st.container(border=True):
        # TODO(4g): replace with the one-sentence RQ3 answer.
        st.markdown(
            "**RQ3 · Which fix keeps it fair most cheaply?**\n\n"
            "Six maintenance strategies compared on 2017, from doing nothing to retraining with a "
            "fairness constraint. *Conclusion coming soon.*"
        )
        st.page_link("pages/4_Fix_Comparison.py", label="Fix Comparison", icon="🛠️")

# ---------------------------------------------------------------- how to use
st.subheader("How to use this app")
st.markdown(
    "1. **Baseline:** how accurate and fair the four models are in 2014, at deployment.\n"
    "2. **Fairness Over Time:** the headline result, the fairness gaps from 2014 to 2017 (RQ1).\n"
    "3. **Why It Changed:** label drift, feature drift, SHAP and per-group error rates (RQ2).\n"
    "4. **Fix Comparison:** six strategies compared on fairness, accuracy and cost (RQ3).\n"
    "5. **Try a Profile:** enter one person and see what each model predicts, why (SHAP), and what changes "
    "if only their sex or race/ethnicity changes."
)
c1, c2 = st.columns(2)
with c1:
    st.page_link("pages/1_Baseline.py", label="Start with the Baseline", icon="▶️")
with c2:
    st.page_link("pages/5_Try_A_Profile.py", label="Or try a profile", icon="🧑")
# ---------------------------------------------------------------- data and method
with st.expander("Data and method"):
    st.markdown(
        "- **Data:** [Folktables](https://github.com/socialfoundations/folktables) ACSIncome task, built "
        "on the US Census American Community Survey (ACS) 1-year person files, **Texas, 2014–2017** "
        "(~127,000–134,000 working adults per year). 2018 is excluded because occupation codes changed "
        "that year; 2020 is avoided because the pandemic disrupted data collection.\n"
        "- **Groups:** sex, race/ethnicity (Hispanic of any race; non-Hispanic White, Black, Asian, Other), "
        "and their 10 intersections.\n"
        "- **Models:** logistic regression and XGBoost, each unmitigated and mitigated (Fairlearn "
        "ExponentiatedGradient, equalized odds on race/ethnicity × sex). Trained on 60% of 2014; "
        "the 2014 numbers use the held-out 20% test set.\n"
        "- **Fairness metrics:** equalized odds difference (EOD, main) and demographic parity difference "
        "(DPD), with 95% bootstrap confidence intervals.\n"
        "- **Diagnosis:** CPI-adjusted label drift, PSI/KS feature drift, SHAP importance over time, "
        "per-group true-positive rates.\n"
        "- **The app never trains.** It only reads results and models saved by the project notebooks."
    )

with st.expander("Limitations"):
    st.markdown(
        "- One state and four years; results may differ elsewhere or over longer periods.\n"
        "- Sex and race/ethnicity are model inputs on purpose, to measure how much the models rely on them. "
        "This is a research model for studying fairness over time, not a deployed decision system.\n"
        "- The smallest groups (NH Asian and NH Other, by sex) have few people earning > $50K, so their "
        "numbers are noisy.\n"
        "- The label uses a fixed $50K line, so part of the change over time is inflation (measured on "
        "Why It Changed)."
    )

# Temporary smoke test (remove at A7)
with st.expander("Artifact check (temporary)"):
    rows = [{"file": name, "rows": len(load_results(path))} for name, path in RESULT_FILES.items()]
    rows.append({"file": "background_sample.csv", "rows": len(load_background())})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.write("Models loaded:", list(get_models()))