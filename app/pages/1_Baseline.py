"""Baseline (2014): how accurate and how fair are the 4 models at deployment?

Reads only precomputed results (2014 rows = held-out test set, 25,408 people):
- metrics_over_time.csv: overall accuracy/AUC, per-group metrics, DPD/EOD gaps
- ci_over_time.csv: 95% bootstrap CIs for DPD/EOD
- label_drift.csv: true share > $50K per group (all 2014 rows)
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.data import CI_PATH, LABEL_DRIFT_PATH, METRICS_PATH
from utils import load_results

st.set_page_config(page_title="Baseline", layout="wide")

# ---------------------------------------------------------------- constants
YEAR = 2014
MODELS = {  # label -> (model, mitigated)
    "LR": ("LR", False),
    "LR mitigated": ("LR", True),
    "XGBoost": ("XGBoost", False),
    "XGBoost mitigated": ("XGBoost", True),
}
ATTRIBUTES = {"Race/ethnicity × sex": "race_eth_x_sex", "Race/ethnicity": "race_eth", "Sex": "sex"}
VERSION_SCALE = alt.Scale(domain=["Unmitigated", "Mitigated"], range=["#1f77b4", "#ff7f0e"])
SMALL_GROUPS = {"NH Other × Female", "NH Other × Male", "NH Asian × Female", "NH Asian × Male"}

# ---------------------------------------------------------------- data
metrics = load_results(METRICS_PATH)
metrics = metrics[metrics["year"] == YEAR].assign(
    version=lambda d: d["mitigated"].map({False: "Unmitigated", True: "Mitigated"}))
ci = load_results(CI_PATH)
ci = ci[ci["year"] == YEAR]
label = load_results(LABEL_DRIFT_PATH)
true_rates = label[(label["year"] == YEAR) & (label["metric"] == "fixed_rate")]


def headline_table(attribute):
    """One row per model: accuracy, AUC (overall) and DPD/EOD with 95% CI (chosen attribute)."""
    rows = []
    for name, (model, mitigated) in MODELS.items():
        m = metrics[(metrics["model"] == model) & (metrics["mitigated"] == mitigated)]
        overall = m[m["attribute"] == "overall"].set_index("metric")["value"]
        gaps = ci[(ci["model"] == model) & (ci["mitigated"] == mitigated)
                  & (ci["attribute"] == attribute)].set_index("metric")
        row = {"Model": name, "Accuracy": overall["accuracy"], "AUC": overall["auc"]}
        for metric in ["DPD", "EOD"]:
            g = gaps.loc[metric]
            row[metric] = g["value"]
            row[f"{metric} 95% CI"] = f"[{g['ci_low']:.3f}, {g['ci_high']:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def group_metric(model, attribute, metric):
    """One per-group metric (e.g. TPR) for both versions (unmitigated / mitigated) of a model."""
    return metrics[(metrics["model"] == model) & (metrics["attribute"] == attribute)
                   & (metrics["metric"] == metric) & (metrics["group"] != "(gap)")]


def tpr_chart(model, attribute):
    """TPR per group, unmitigated vs. mitigated, sorted by the unmitigated TPR."""
    df = group_metric(model, attribute, "TPR")
    df = df.assign(label=df["group"].where(~df["group"].isin(SMALL_GROUPS), df["group"] + " *"))
    order = df[df["version"] == "Unmitigated"].sort_values("value")["label"].tolist()
    return (
        alt.Chart(df)
        .mark_point(size=130, filled=True, opacity=1)
        .encode(
            x=alt.X("value:Q", title="True-positive rate (share of > $50K earners the model catches)",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[i / 10 for i in range(11)], format=".1f")),
            y=alt.Y("label:N", title=None, sort=order, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("version:N", title=None, scale=VERSION_SCALE, legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("version:N", title="Version"),
                     alt.Tooltip("value:Q", title="TPR", format=".3f")],
        )
        .properties(height=34 * df["label"].nunique() + 40)
    )


def accuracy_vs_tpr(model, mitigated):
    """Per race/ethnicity group: accuracy and TPR side by side (accuracy hides the TPR gap)."""
    df = metrics[(metrics["model"] == model) & (metrics["mitigated"] == mitigated)
                 & (metrics["attribute"] == "race_eth") & (metrics["metric"].isin(["Accuracy", "TPR"]))]
    df = df.assign(measure=df["metric"].map({"Accuracy": "Accuracy", "TPR": "True-positive rate"}))
    order = df[df["metric"] == "TPR"].sort_values("value")["group"].tolist()
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("group:N", title=None, sort=order),
            yOffset=alt.YOffset("measure:N"),
            color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(range=["#9e9e9e", "#d62728"])),
            tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("measure:N", title="Measure"),
                     alt.Tooltip("value:Q", title="Value", format=".3f")],
        )
        .properties(height=300)
    )


def sex_gap(model, mitigated):
    """True vs. predicted share > $50K by sex (true = all 2014 rows; predicted = test-set selection rate)."""
    true = true_rates[true_rates["attribute"] == "sex"].set_index("group")["value"]
    pred = metrics[(metrics["model"] == model) & (metrics["mitigated"] == mitigated)
                   & (metrics["attribute"] == "sex") & (metrics["metric"] == "Selection_rate")
                   ].set_index("group")["value"]
    return pd.DataFrame({"True share > $50K": true, "Predicted share > $50K": pred}).loc[["Male", "Female"]]


# ---------------------------------------------------------------- page
st.title("Baseline (2014)")
st.markdown(
    "**At deployment, the unmitigated models are accurate but unfair, and fairness mitigation fixes "
    "most of the gap for about one point of accuracy.** All numbers on this page come from the 2014 "
    "held-out test set (25,408 people), used once, after every modelling choice was made."
)

with st.expander("Setup: data, models and mitigation"):
    st.markdown(
        "- **Data:** Folktables ACSIncome, Texas, 2014 ACS 1-year person file: 127,039 working adults. "
        "Label: personal income > $50,000 (31.6% of people).\n"
        "- **Features (10):** age, class of worker, education, marital status, occupation, place of birth, "
        "household role, hours worked, sex, race/ethnicity. Race/ethnicity combines race and Hispanic "
        "origin into 5 groups (Hispanic of any race; non-Hispanic White, Black, Asian, Other).\n"
        "- **Split:** 60/20/20 train/validation/test, stratified by label × race/ethnicity × sex.\n"
        "- **Models:** logistic regression and XGBoost, each in two versions: unmitigated, and mitigated "
        "with Fairlearn's ExponentiatedGradient under an **equalized odds** constraint on the 10 "
        "race/ethnicity × sex groups (thresholded at 0.5).\n"
        "- These four models are then **frozen**: the later pages run them on 2015–2017 without retraining."
    )

attr_label = st.radio("Groups compared", list(ATTRIBUTES), horizontal=True)
attribute = ATTRIBUTES[attr_label]
table = headline_table(attribute)

st.subheader(f"Accuracy and fairness — {attr_label}")
st.dataframe(
    table.style.format({"Accuracy": "{:.3f}", "AUC": "{:.3f}", "DPD": "{:.3f}", "EOD": "{:.3f}"}),
    hide_index=True,
)
eod = table.set_index("Model")["EOD"]
acc = table.set_index("Model")["Accuracy"]
c1, c2 = st.columns(2)
for col, model in zip((c1, c2), ("LR", "XGBoost")):
    col.metric(
        f"{model}: EOD after mitigation",
        f"{eod[f'{model} mitigated']:.3f}",
        f"{(eod[f'{model} mitigated'] / eod[model] - 1) * 100:.0f}% vs. unmitigated "
        f"(accuracy {(acc[f'{model} mitigated'] - acc[model]) * 100:+.1f} points)",
        delta_color="inverse",
    )
st.caption(
    "EOD = largest gap between groups in true- or false-positive rate; DPD = largest gap in the share "
    "predicted > $50K (lower = fairer for both). Mitigated AUC comes from a vote share and is not "
    "comparable. The mitigated models' EOD intervals are wide and lopsided: EOD is a maximum over many "
    "noisy group gaps, so resampling pushes it upward. Read the intervals by width and overlap."
)

# ---------------------------------------------------------------- per-group TPR
st.subheader("Who does the model miss?")
model = st.radio("Model", ["LR", "XGBoost"], horizontal=True, index=1)
st.altair_chart(tpr_chart(model, attribute))
st.caption(
    "Unmitigated (blue) vs. mitigated (orange) true-positive rate per group in 2014. Mitigation raises "
    "the groups the model missed most and lowers the top groups, pulling them together. "
    "* Small groups (about 40–60 people earning > $50K in the test set): noisy."
)

# ---------------------------------------------------------------- accuracy hides unfairness
st.subheader("Accuracy hides the unfairness")
c1, c2 = st.columns([3, 2])
with c1:
    st.altair_chart(accuracy_vs_tpr(model, False))
with c2:
    race = metrics[(metrics["model"] == model) & (~metrics["mitigated"]) & (metrics["attribute"] == "race_eth")]
    tpr = race[race["metric"] == "TPR"].set_index("group")["value"]
    accuracy = race[race["metric"] == "Accuracy"].set_index("group")["value"]
    st.markdown(
        f"For unmitigated {model}, **Hispanic workers have one of the highest accuracies "
        f"({accuracy['Hispanic']:.0%}) but the lowest true-positive rate ({tpr['Hispanic']:.0%})**: "
        f"the model misses {1 - tpr['Hispanic']:.0%} of Hispanic Texans who earn over $50K. "
        "In a group where few people earn over $50K, predicting \"no\" is usually right, so accuracy "
        "stays high while the people who do earn more are missed. That is why this project measures "
        "fairness with error-rate gaps, not accuracy."
    )

# ---------------------------------------------------------------- sex gap
st.subheader("The model widens the real sex gap")
gap = sex_gap(model, False)
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(gap.style.format("{:.1%}"))
with c2:
    true_gap = (gap.loc["Male", "True share > $50K"] - gap.loc["Female", "True share > $50K"]) * 100
    pred_gap = (gap.loc["Male", "Predicted share > $50K"] - gap.loc["Female", "Predicted share > $50K"]) * 100
    st.markdown(
        f"In the data, men are {true_gap:.1f} points more likely than women to earn over $50K. "
        f"Unmitigated {model} predicts a gap of **{pred_gap:.1f} points**: it exaggerates a real "
        "difference rather than just reflecting it."
    )
st.caption(
    "True share: all 2014 rows. Predicted share: 2014 test set. Both samples come from the same "
    "stratified split, so their true shares match closely."
)

st.divider()
st.markdown(
    "**Next:** these four models are frozen and run on 2015–2017. "
    "Does this baseline hold? See **Fairness Over Time**."
)