"""Fairness Over Time (headline page, RQ1).

Reads only precomputed results:
- ci_over_time.csv: DPD / EOD per year, model and attribute, with 95% bootstrap CIs
- metrics_over_time.csv: overall accuracy and AUC per year and model
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.data import CI_PATH, METRICS_PATH
from utils import load_results

st.set_page_config(page_title="Fairness Over Time", layout="wide")

# ---------------------------------------------------------------- constants
ATTRIBUTES = {  # radio label -> value in the CSV `attribute` column
    "Race/ethnicity × sex": "race_eth_x_sex",
    "Race/ethnicity": "race_eth",
    "Sex": "sex",
}
METRIC_NAMES = {
    "EOD": "Equalized odds difference",
    "DPD": "Demographic parity difference",
}
VERSION_ORDER = ["Unmitigated", "Mitigated"]
VERSION_COLORS = ["#1f77b4", "#ff7f0e"]  # blue / orange, same as the notebook PNG
MODEL_ORDER = ["LR", "XGBoost"]
FIRST_YEAR, LAST_YEAR = 2014, 2017


def add_version(df):
    """Add a readable 'version' column (Unmitigated / Mitigated) from the bool `mitigated`."""
    df = df.copy()
    df["version"] = df["mitigated"].map({False: "Unmitigated", True: "Mitigated"})
    return df


# ---------------------------------------------------------------- data
ci = add_version(load_results(CI_PATH))
metrics = load_results(METRICS_PATH)
overall = add_version(metrics[metrics["attribute"] == "overall"])  # accuracy, auc


def gap_data(metric, attribute):
    """Rows for one gap metric and one attribute: 4 models × 4 years = 16 rows."""
    return ci[(ci["metric"] == metric) & (ci["attribute"] == attribute)]


def gap_chart(df, metric):
    """Line + 95% CI band per model version, one panel per model, shared y-axis from 0."""
    y_max = float(df["ci_high"].max()) * 1.1
    color = alt.Color(
        "version:N",
        title=None,
        scale=alt.Scale(domain=VERSION_ORDER, range=VERSION_COLORS),
        legend=alt.Legend(orient="top", symbolOpacity=1),
    )
    x = alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0))
    tooltip = [
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("version:N", title="Version"),
        alt.Tooltip("year:O", title="Year"),
        alt.Tooltip("value:Q", title=metric, format=".3f"),
        alt.Tooltip("ci_low:Q", title="95% CI low", format=".3f"),
        alt.Tooltip("ci_high:Q", title="95% CI high", format=".3f"),
    ]

    base = alt.Chart().encode(x=x, color=color)
    band = base.mark_area(opacity=0.3).encode(
        y=alt.Y("ci_low:Q", title=metric, scale=alt.Scale(domain=[0, y_max])),
        y2="ci_high:Q",
    )
    line = base.mark_line(point=True).encode(
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, y_max])),
        tooltip=tooltip,
    )
    return (
        alt.layer(band, line, data=df)
        .properties(width=380, height=300)
        .facet(column=alt.Column("model:N", title=None, sort=MODEL_ORDER))
    )


def change_cards(df, metric):
    """One st.metric per model version: 2017 value and change since 2014 (a rise is bad)."""
    cols = st.columns(4)
    i = 0
    for model in MODEL_ORDER:
        for version in VERSION_ORDER:
            rows = df[(df["model"] == model) & (df["version"] == version)].set_index("year")
            first, last = rows.loc[FIRST_YEAR, "value"], rows.loc[LAST_YEAR, "value"]
            cols[i].metric(
                label=f"{model} · {version.lower()}",
                value=f"{last:.3f}",
                delta=f"{last - first:+.3f} since {FIRST_YEAR}",
                delta_color="inverse",  # higher gap = worse, so show rises in red
            )
            i += 1


def accuracy_auc_chart(df):
    """Change in accuracy and AUC since 2014 (percentage points), shared y-axis.

    Plotting the change (not the level) on one axis makes the two metrics directly
    comparable: separate zoomed axes would exaggerate or hide differences.
    Colors match the gap chart (version); dash pattern = model.
    """
    base_year = df[df["year"] == FIRST_YEAR].set_index(["model", "version", "metric"])["value"]
    df = df.assign(
        change=(df["value"] - df.set_index(["model", "version", "metric"]).index.map(base_year)) * 100,
        metric_label=df["metric"].map({"accuracy": "Accuracy", "auc": "AUC"}),
    )
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("change:Q", title=f"Change since {FIRST_YEAR} (points)"),
            color=alt.Color(
                "version:N",
                title=None,
                scale=alt.Scale(domain=VERSION_ORDER, range=VERSION_COLORS),
                legend=alt.Legend(orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "model:N",
                title=None,
                sort=MODEL_ORDER,
                legend=None,  # explained in the caption below the chart
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("version:N", title="Version"),
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("value:Q", title="Value", format=".3f"),
                alt.Tooltip("change:Q", title="Change (points)", format="+.2f"),
            ],
        )
        .properties(width=380, height=250)
        .facet(
            column=alt.Column(
                "metric_label:N", title=None, sort=["Accuracy", "AUC"]
            )
        )
    )


# ---------------------------------------------------------------- page
st.title("Fairness Over Time")
st.markdown(
    "**Does a model that is fair in 2014 stay fair through 2017 without retraining?**  \n"
    "Fairness mostly holds over three years, but it **erodes gradually and unevenly**. "
    "No model collapses, and mitigation's benefit survives: in 2017 the mitigated models' "
    "gaps are still far below the unmitigated ones."
)

attr_label = st.radio("Groups compared", list(ATTRIBUTES), horizontal=True)
attribute = ATTRIBUTES[attr_label]

for tab, metric in zip(st.tabs(["EOD (main metric)", "DPD"]), ["EOD", "DPD"]):
    with tab:
        data = gap_data(metric, attribute)
        st.subheader(f"{METRIC_NAMES[metric]} by year — {attr_label}")
        st.altair_chart(gap_chart(data, metric))
        st.caption(
            "Lines: point estimates. Shaded bands: 95% bootstrap confidence intervals. "
            "Lower is fairer (0 = no gap)."
        )
        st.markdown(f"**{LAST_YEAR} value and change since {FIRST_YEAR}** (red = gap widened)")
        change_cards(data, metric)

st.divider()
st.subheader("Accuracy falls, AUC stays flat")
st.altair_chart(accuracy_auc_chart(overall))
st.markdown(
    "Every model loses about 1.2 points of accuracy by 2017, while AUC falls far less "
    "(0.3–0.5 points for the unmitigated models). "
    "The models still **rank** people correctly, but the 2014 decision cutoff is out of date: "
    "more people with the same features now earn over $50K. "
    "See **Why It Changed** for the diagnosis."
)
st.caption(
    "Solid lines: LR. Dashed lines: XGBoost. "
    "Mitigated AUC is computed from a vote share, not a calibrated probability, "
    "so it is not directly comparable with the unmitigated models' AUC."
)

with st.expander("How to read these metrics"):
    st.markdown(
        "- **EOD (equalized odds difference):** the largest gap between groups in "
        "true-positive rate (people earning > $50K who are correctly flagged) or "
        "false-positive rate. It is the main metric because the mitigation targeted it.\n"
        "- **DPD (demographic parity difference):** the largest gap between groups in the "
        "share predicted > $50K. It also moves when the real share of high earners changes "
        "in a group, so a rising DPD is not automatically a model failure.\n"
        "- **Race/ethnicity × sex** compares the 10 intersectional groups "
        "(e.g. Hispanic × Female); it is the headline view."
    )

with st.expander("Caveats"):
    st.markdown(
        "- **2014 is measured on the held-out test set** (25,408 people); 2015–2017 use every "
        "person in that year (~130,000). So 2014's bands are the widest.\n"
        "- 2014 intervals use 200 bootstrap resamples, 2015–2017 use 100 (for runtime). "
        "This affects how stable the bounds are, not their expected width.\n"
        "- Overlapping bands do not prove there is no change. Changes smaller than about "
        "twice their standard error should be treated as unconvincing.\n"
        "- Mitigated models use deterministic (thresholded) predictions, so Fairlearn's "
        "fairness guarantee (stated for randomized predictions) does not formally apply."
    )