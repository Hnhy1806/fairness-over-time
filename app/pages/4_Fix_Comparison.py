"""Fix Comparison (RQ3): which maintenance strategy restores fairness most cheaply?

Reads only fix_comparison.csv (Phase 4): every strategy fitted on 2016, evaluated on all of 2017.
Columns: strategy, model, attribute, group, metric, value (fit time = metric fit_seconds).
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.data import FIX_PATH
from utils import load_results

st.set_page_config(page_title="Fix Comparison", layout="wide")

# ---------------------------------------------------------------- constants
STRATEGIES = {  # CSV label -> short name shown in the app (order = order in tables and legends)
    "none": "S0 · Do nothing",
    "none_mitigated": "S0m · Keep 2014 mitigated model",
    "global_cutoff": "S1 · New single cutoff",
    "group_cutoffs": "S2 · Group-specific cutoffs",
    "retrain": "S3 · Retrain",
    "retrain_mitigate": "S4 · Retrain + re-mitigate",
}
DESCRIPTIONS = {
    "none": "Keep the frozen 2014 model and its 0.5 cutoff.",
    "none_mitigated": "Keep the frozen 2014 fairness-mitigated model (ExponentiatedGradient, equalized odds).",
    "global_cutoff": "Keep the 2014 model; pick one new cutoff that maximizes accuracy on 2016.",
    "group_cutoffs": "Keep the 2014 model; fit a separate cutoff per race/ethnicity × sex group on 2016 "
                     "(Fairlearn ThresholdOptimizer, equalized odds).",
    "retrain": "Refit preprocessing and model on 2016, same settings.",
    "retrain_mitigate": "Refit on 2016 with the fairness constraint (ExponentiatedGradient, equalized odds).",
}
NEEDS_2016_LABELS = {"none": "No", "none_mitigated": "No", "global_cutoff": "Yes", "group_cutoffs": "Yes",
                     "retrain": "Yes", "retrain_mitigate": "Yes"}
NEW_MODEL = {"none": "No", "none_mitigated": "No", "global_cutoff": "No", "group_cutoffs": "No",
             "retrain": "Yes", "retrain_mitigate": "Yes"}
DECISION_RULE = {
    "none": "One cutoff",
    "none_mitigated": "Ensemble vote (uses sex/race as inputs)",
    "global_cutoff": "One cutoff",
    "group_cutoffs": "Separate cutoff per group, randomized near the cutoff",
    "retrain": "One cutoff",
    "retrain_mitigate": "Ensemble vote (uses sex/race as inputs)",
}
ATTRIBUTES = {"Race/ethnicity × sex": "race_eth_x_sex", "Race/ethnicity": "race_eth", "Sex": "sex"}
MODEL_ORDER = ["LR", "XGBoost"]
NOISE_EOD = 0.04  # 2017 EOD intervals are about ±0.02, so treat differences < ~0.04 as noise
SMALL_GROUPS = {"NH Other × Female", "NH Other × Male", "NH Asian × Female", "NH Asian × Male"}
STRATEGY_COLORS = alt.Scale(
    domain=list(STRATEGIES.values()),
    range=["#7f7f7f", "#ff7f0e", "#9467bd", "#2ca02c", "#1f77b4", "#d62728"],
)

# ---------------------------------------------------------------- data
fix = load_results(FIX_PATH)
fix = fix.assign(strategy_name=fix["strategy"].map(STRATEGIES))


def summary(attribute):
    """One row per model × strategy: accuracy, AUC, fit time (overall) + DPD, EOD (chosen attribute)."""
    overall = fix[fix["attribute"] == "overall"].pivot_table(
        index=["model", "strategy"], columns="metric", values="value")
    gaps = fix[(fix["attribute"] == attribute) & (fix["group"] == "(gap)")].pivot_table(
        index=["model", "strategy"], columns="metric", values="value")
    out = overall.join(gaps).reset_index()
    out["strategy_name"] = out["strategy"].map(STRATEGIES)
    out["order"] = out["strategy"].map({s: i for i, s in enumerate(STRATEGIES)})
    return out.sort_values(["model", "order"]).drop(columns="order")


def tradeoff_chart(df):
    """EOD (x) vs. accuracy (y): one point per model × strategy. Best = top-left."""
    x_scale = alt.Scale(domain=[0, max(0.65, float(df["EOD"].max()) * 1.05)])
    y = alt.Y("accuracy:Q", title="Accuracy in 2017", scale=alt.Scale(zero=False))
    color = alt.Color("strategy_name:N", title="Strategy", scale=STRATEGY_COLORS,
                      sort=list(STRATEGIES.values()), legend=alt.Legend(symbolOpacity=1))
    df = df.assign(eod_low=df["EOD"] - NOISE_EOD / 2, eod_high=df["EOD"] + NOISE_EOD / 2)

    # EOD noise band around each point (±0.02): overlapping bars = tied on fairness
    noise = alt.Chart(df).mark_rule(opacity=0.35, strokeWidth=3).encode(
        x=alt.X("eod_low:Q", scale=x_scale), x2="eod_high:Q", y=y, color=color)
    points = alt.Chart(df).mark_point(size=160, filled=True, opacity=0.9).encode(
        x=alt.X("EOD:Q", title="Equalized odds difference in 2017 (lower = fairer)", scale=x_scale),
        y=y,
        color=color,
        shape=alt.Shape("model:N", title="Model", sort=MODEL_ORDER),
        tooltip=[
            alt.Tooltip("strategy_name:N", title="Strategy"), alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("accuracy:Q", title="Accuracy", format=".4f"),
            alt.Tooltip("EOD:Q", title="EOD", format=".3f"), alt.Tooltip("DPD:Q", title="DPD", format=".3f"),
            alt.Tooltip("fit_seconds:Q", title="Fit time (s)", format=",.1f"),
        ],
    )
    return (noise + points).properties(height=380)


def tpr_chart(model, strategies, attribute):
    """TPR per group for the chosen strategies (one model), in 2017."""
    df = fix[(fix["model"] == model) & (fix["attribute"] == attribute) & (fix["metric"] == "TPR")
             & (fix["group"] != "(gap)") & (fix["strategy"].isin(strategies))]
    df = df.assign(label=df["group"].where(~df["group"].isin(SMALL_GROUPS), df["group"] + " *"))
    order = (df[df["strategy"] == strategies[0]].sort_values("value")["label"].tolist()
             if strategies else None)
    return (
        alt.Chart(df)
        .mark_point(size=120, filled=True, opacity=0.9)
        .encode(
            x=alt.X("value:Q", title="True-positive rate in 2017", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(values=[i / 10 for i in range(11)], format=".1f")),
            y=alt.Y("label:N", title=None, sort=order, axis=alt.Axis(labelLimit=220)),
            color=alt.Color("strategy_name:N", title="Strategy", scale=STRATEGY_COLORS,
                            sort=list(STRATEGIES.values()), legend=alt.Legend(orient="top", symbolOpacity=1)),
            tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("strategy_name:N", title="Strategy"),
                     alt.Tooltip("value:Q", title="TPR", format=".3f")],
        )
        .properties(height=36 * df["label"].nunique() + 40)
    )


# ---------------------------------------------------------------- page
st.title("Fix Comparison")
st.markdown(
    "**Which maintenance strategy keeps the model fair most cheaply?**  \n"
    "Scenario: it is the end of 2016 and the 2016 labels are available. Each strategy is fitted on 2016 "
    "and evaluated on **all of 2017**, a year none of them has seen."
)

# TODO(4g): replace this placeholder with the RQ3 answer (one sentence + 2–3 supporting points).
st.info("**Conclusion:** coming soon (written together with the RQ3 answer in notebook 04).", icon="📝")

with st.expander("The six strategies", expanded=False):
    st.dataframe(
        pd.DataFrame({"Strategy": STRATEGIES.values(), "What it does": DESCRIPTIONS.values()}),
        hide_index=True,
    )

attr_label = st.radio("Groups compared", list(ATTRIBUTES), horizontal=True)
attribute = ATTRIBUTES[attr_label]
table = summary(attribute)

# ---------------------------------------------------------------- trade-off chart
st.subheader(f"Fairness vs. accuracy in 2017 — {attr_label}")
st.altair_chart(tradeoff_chart(table))
st.caption(
    "Each point is one strategy (color) for one model (shape). Best = **top-left** (fair and accurate). "
    f"Horizontal bars show ±{NOISE_EOD / 2:.2f}, the approximate noise in 2017 EOD: when two bars overlap, "
    "treat the strategies as tied on fairness. Mitigated AUC is not used here (it comes from a vote share)."
)

# ---------------------------------------------------------------- full table
st.subheader("All results")
for model in MODEL_ORDER:
    rows = table[table["model"] == model]
    shown = pd.DataFrame({
        "Strategy": rows["strategy_name"],
        "Accuracy": rows["accuracy"],
        "AUC": rows["auc"],
        "DPD": rows["DPD"],
        "EOD": rows["EOD"],
        "Fit time (s)": rows["fit_seconds"],
        "Needs 2016 labels": rows["strategy"].map(NEEDS_2016_LABELS),
        "New model to deploy": rows["strategy"].map(NEW_MODEL),
        "Decision rule": rows["strategy"].map(DECISION_RULE),
    })
    st.markdown(f"**{model}**")
    st.dataframe(
        shown.style.format({"Accuracy": "{:.4f}", "AUC": "{:.4f}", "DPD": "{:.3f}", "EOD": "{:.3f}",
                            "Fit time (s)": "{:,.1f}"})
            .highlight_min(subset=["EOD", "DPD"], color="rgba(44, 160, 44, 0.35)")
            .highlight_max(subset=["Accuracy"], color="rgba(44, 160, 44, 0.35)"),
        hide_index=True,
    )
st.caption(
    "Green = best value in the column for that model. DPD and EOD use the groups chosen above; accuracy, "
    "AUC and fit time are overall. Mitigated AUC (S0m, S4) is computed from a vote share and is not "
    "comparable with the others. Fit times are from one laptop: read them as orders of magnitude "
    "(under 1 s, seconds, minutes)."
)

# ---------------------------------------------------------------- per-group TPR
st.subheader("Who gets caught? True-positive rate by group")
c1, c2 = st.columns([1, 3])
with c1:
    model = st.radio("Model", MODEL_ORDER, horizontal=True)
with c2:
    chosen = st.multiselect(
        "Strategies", list(STRATEGIES), default=["none", "group_cutoffs", "retrain_mitigate"],
        format_func=STRATEGIES.get,
    )
if chosen:
    st.altair_chart(tpr_chart(model, chosen, attribute))
    st.caption(
        "Groups are sorted by the first selected strategy's TPR. A fair strategy pulls the points of all "
        "groups close together. * Small groups: ignore differences below ~0.05."
    )
else:
    st.warning("Select at least one strategy.")

with st.expander("Limitations"):
    st.markdown(
        "- **Best case for every fix:** the 2016 labels are assumed available at the end of 2016. In practice "
        "the Census publishes a survey year's data about a year later (label delay), so a real system "
        "would be fixing an older year.\n"
        "- **One fit year, one test year, point estimates only** (no bootstrap): treat EOD differences "
        f"below ~{NOISE_EOD} as noise.\n"
        "- **S2 gives randomized decisions** near each group's cutoff (two identical people can get "
        "different outcomes; fixed seed here) and applies an explicit group-specific rule.\n"
        "- **S0m and S4 use thresholded votes**, so Fairlearn's fairness guarantee (stated for randomized "
        "predictions) does not formally apply. They also use sex and race/ethnicity directly as inputs "
        "(see the what-if grid on **Try a Profile**).\n"
        "- **Retraining (S3, S4) used all 131,089 people of 2016**, 1.7× the 2014 training set, so more "
        "occupation and birthplace categories got their own column.\n"
        "- **EOD is a maximum gap**, so the smallest group can decide it (e.g. LR S4's EOD is set by "
        "NH Other × Female, 1,239 people)."
    )