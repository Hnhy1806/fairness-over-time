"""Why It Changed (RQ2): diagnose why fairness drifts in the frozen 2014 models.

Reads only precomputed Phase 3 results:
- label_drift.csv: share > $50K per year, fixed $50K line vs. CPI-indexed line
- drift_over_time.csv: PSI / KS per feature (baseline = 2014 training rows; year 2014 = noise floor)
- shap_over_time.csv: mean |SHAP| per feature, year and model (unmitigated LR + XGBoost)
- metrics_over_time.csv: per-group TPR (race/ethnicity × sex)
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.data import DRIFT_PATH, LABEL_DRIFT_PATH, METRICS_PATH, SHAP_PATH
from utils import load_results

st.set_page_config(page_title="Why It Changed", layout="wide")

# ---------------------------------------------------------------- constants
FIRST_YEAR, LAST_YEAR = 2014, 2017
FEATURE_NAMES = {
    "AGEP": "Age", "COW": "Class of worker", "SCHL": "Education", "MAR": "Marital status",
    "OCCP": "Occupation", "POBP": "Place of birth", "RELP": "Household role",
    "WKHP": "Hours worked", "SEX": "Sex", "race_eth": "Race/ethnicity",
}
PSI_SMALL = 0.1  # common rule of thumb: PSI < 0.1 = small shift
SMALL_GROUPS = {"NH Other × Female", "NH Other × Male", "NH Asian × Female", "NH Asian × Male"}
MODEL_CHOICES = {  # radio label -> (model, mitigated)
    "LR": ("LR", False),
    "XGBoost": ("XGBoost", False),
    "LR mitigated": ("LR", True),
    "XGBoost mitigated": ("XGBoost", True),
}
YEAR_COLORS = alt.Scale(domain=[str(FIRST_YEAR), str(LAST_YEAR)], range=["#9e9e9e", "#d62728"])

# ---------------------------------------------------------------- data
label = load_results(LABEL_DRIFT_PATH)
drift = load_results(DRIFT_PATH)
shap = load_results(SHAP_PATH)
metrics = load_results(METRICS_PATH)


def label_split():
    """Per group, 2014 -> 2017 rise in the share > $50K, split into inflation and real growth (points).

    total = fixed(2017) - fixed(2014); inflation = fixed(2017) - indexed(2017);
    real = indexed(2017) - indexed(2014). inflation + real = total.
    """
    wide = label.pivot(index=["attribute", "group", "year"], columns="metric", values="value").reset_index()
    first = wide[wide["year"] == FIRST_YEAR].set_index(["attribute", "group"])
    last = wide[wide["year"] == LAST_YEAR].set_index(["attribute", "group"])
    out = pd.DataFrame({
        f"{FIRST_YEAR} share": first["fixed_rate"] * 100,
        f"{LAST_YEAR} share": last["fixed_rate"] * 100,
        "Total rise": (last["fixed_rate"] - first["fixed_rate"]) * 100,
        "Inflation part": (last["fixed_rate"] - last["indexed_rate"]) * 100,
        "Real-growth part": (last["indexed_rate"] - first["indexed_rate"]) * 100,
    })
    out["Inflation share"] = out["Inflation part"] / out["Total rise"] * 100
    out = out.reset_index()
    out["Group"] = out["group"].where(out["attribute"] != "overall", "Everyone")
    order = {"overall": 0, "sex": 1, "race_eth": 2}
    return out.sort_values(["attribute", "Total rise"], key=lambda s: s.map(order) if s.name == "attribute" else -s)


def label_chart():
    """Overall share > $50K per year: fixed $50K line vs. inflation-indexed line."""
    df = label[label["attribute"] == "overall"].assign(
        line=lambda d: d["metric"].map({"fixed_rate": "Fixed $50K line (the label)",
                                        "indexed_rate": "Inflation-adjusted line"}),
        pct=lambda d: d["value"] * 100,
    )
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("pct:Q", title="Share earning > the line (%)", scale=alt.Scale(zero=False)),
            color=alt.Color("line:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(range=["#d62728", "#1f77b4"])),
            tooltip=[alt.Tooltip("line:N", title="Line"), alt.Tooltip("year:O", title="Year"),
                     alt.Tooltip("pct:Q", title="Share (%)", format=".1f")],
        )
        .properties(height=280)
    )


def psi_chart():
    """2017 PSI per feature (bars) with the 2014 noise floor (ticks) and the 0.1 'small' line."""
    psi = drift[drift["metric"] == "PSI"].assign(feature_name=lambda d: d["feature"].map(FEATURE_NAMES))
    last = psi[psi["year"] == LAST_YEAR]
    noise = psi[psi["year"] == FIRST_YEAR]
    order = last.sort_values("value", ascending=False)["feature_name"].tolist()
    x = alt.X("value:Q", title="PSI (2017 vs. 2014 training data)", scale=alt.Scale(domain=[0, 0.11]))
    y = alt.Y("feature_name:N", title=None, sort=order)
    bars = alt.Chart(last).mark_bar(color="#1f77b4").encode(
        x=x, y=y, tooltip=[alt.Tooltip("feature_name:N", title="Feature"),
                           alt.Tooltip("value:Q", title="PSI 2017", format=".4f")])
    ticks = alt.Chart(noise).mark_tick(color="orange", thickness=2, size=18).encode(
        x="value:Q", y=y, tooltip=[alt.Tooltip("feature_name:N", title="Feature"),
                                   alt.Tooltip("value:Q", title="Noise floor (2014 test vs. train)", format=".4f")])
    rule = alt.Chart(pd.DataFrame({"x": [PSI_SMALL]})).mark_rule(color="gray", strokeDash=[4, 4]).encode(x="x:Q")
    rule_text = alt.Chart(pd.DataFrame({"x": [PSI_SMALL], "t": ["0.1 = 'small' threshold"]})).mark_text(
        align="right", dx=-4, dy=-6, color="gray").encode(x="x:Q", y=alt.value(0), text="t:N")
    return (bars + ticks + rule + rule_text).properties(height=320)


def ks_table():
    """KS statistic for the ordered features per year, with the noise-floor ratio for 2017."""
    ks = drift[drift["metric"] == "KS"].pivot(index="feature", columns="year", values="value")
    ks.columns = ["2014 (noise floor)" if c == FIRST_YEAR else str(c) for c in ks.columns]
    ks[f"{LAST_YEAR} / noise"] = ks[str(LAST_YEAR)] / ks["2014 (noise floor)"]
    ks.index = ks.index.map(FEATURE_NAMES)
    return ks.sort_values(f"{LAST_YEAR} / noise", ascending=False)


def shap_chart():
    """Mean |SHAP| per feature over the years, one panel per model."""
    df = shap[shap["metric"] == "mean_abs_shap"].assign(feature_name=lambda d: d["feature"].map(FEATURE_NAMES))
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="Mean |SHAP| (log-odds)"),
            color=alt.Color("feature_name:N", title="Feature", scale=alt.Scale(scheme="tableau10")),
            tooltip=[alt.Tooltip("feature_name:N", title="Feature"), alt.Tooltip("year:O", title="Year"),
                     alt.Tooltip("value:Q", title="Mean |SHAP|", format=".3f")],
        )
        .properties(width=360, height=300)
        .facet(column=alt.Column("model:N", title=None, sort=["LR", "XGBoost"]))
    )


def tpr_data(model, mitigated):
    """TPR per race/ethnicity × sex group in 2014 and 2017 for one model."""
    rows = metrics[
        (metrics["attribute"] == "race_eth_x_sex") & (metrics["metric"] == "TPR")
        & (metrics["model"] == model) & (metrics["mitigated"] == mitigated)
        & (metrics["year"].isin([FIRST_YEAR, LAST_YEAR]))
    ]
    wide = rows.pivot(index="group", columns="year", values="value")
    wide.columns = [str(c) for c in wide.columns]  # Altair needs string column names
    wide["change"] = wide[str(LAST_YEAR)] - wide[str(FIRST_YEAR)]
    wide = wide.reset_index()
    wide["label"] = wide["group"].where(~wide["group"].isin(SMALL_GROUPS), wide["group"] + " *")
    return wide.sort_values("change")


def tpr_chart(wide):
    """Dumbbell chart: each group's TPR in 2014 (gray) and 2017 (red), sorted by change."""
    order = wide["label"].tolist()
    long = wide.melt(id_vars=["label", "change"], value_vars=[str(FIRST_YEAR), str(LAST_YEAR)],
                     var_name="year", value_name="tpr")
    y = alt.Y("label:N", title=None, sort=order, axis=alt.Axis(labelLimit=220))
    lines = alt.Chart(wide).mark_rule(color="gray", opacity=0.6).encode(
        x=alt.X(f"{FIRST_YEAR}:Q"), x2=alt.X2(f"{LAST_YEAR}:Q"), y=y)
    points = alt.Chart(long).mark_circle(size=110, opacity=1).encode(
        x=alt.X("tpr:Q", title="True-positive rate (share of > $50K earners the model catches)",
                scale=alt.Scale(domain=[0, 1])),
        y=y,
        color=alt.Color("year:N", title=None, scale=YEAR_COLORS, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("year:N", title="Year"),
                 alt.Tooltip("tpr:Q", title="TPR", format=".3f"),
                 alt.Tooltip("change:Q", title="Change 2014→2017", format="+.3f")],
    )
    return (lines + points).properties(height=360)


# ---------------------------------------------------------------- page
st.title("Why It Changed")
st.markdown(
    "**Answer: concept drift meets a fixed cutoff.** The people the models see barely change, and the "
    "models use their features the same way every year. What changes is the outcome: more people with "
    "the same features now earn over $50K, mostly through real income growth. The 2014 cutoff misses "
    "many of these new high earners, and it misses them **unevenly** across groups, which widens the "
    "fairness gap."
)

# 1 ------------------------------------------------------------ label drift
st.header("1. The label moved")
c1, c2 = st.columns([3, 2])
with c1:
    st.altair_chart(label_chart())
with c2:
    split = label_split()
    overall = split[split["attribute"] == "overall"].iloc[0]
    st.metric(f"Share earning > $50K, {LAST_YEAR}", f"{overall[f'{LAST_YEAR} share']:.1f}%",
              f"{overall['Total rise']:+.1f} points since {FIRST_YEAR}", delta_color="off")
    st.markdown(
        f"Of that rise, **{100 - overall['Inflation share']:.0f}% is real income growth** and only "
        f"**{overall['Inflation share']:.0f}% is inflation** (the fixed $50K line losing value). "
        "So the 2014 models go out of date mainly because the population really changed."
    )
st.caption(
    "Red: the label the models were trained on (income > $50K). Blue: the same line adjusted for "
    "inflation (CPI-U, 2014 dollars). The gap between them is the part caused by inflation alone."
)
with st.expander("By group (percentage points)"):
    show = split[["Group", f"{FIRST_YEAR} share", f"{LAST_YEAR} share", "Total rise",
                  "Inflation part", "Real-growth part", "Inflation share"]]
    st.dataframe(
        show.style.format({c: "{:.1f}" for c in show.columns if c != "Group"})
            .format({"Inflation share": "{:.0f}%"}),
        hide_index=True,
    )
    st.caption(
        "Shares are in %, changes in percentage points. Women's real growth (+3.8) outpaces men's (+2.6), "
        "so the true male–female gap narrows. Hispanic workers had the slowest real growth of any group. "
        "NH Other (~2,200 people per year) is too small to interpret."
    )

# 2 ------------------------------------------------------------ feature drift
st.header("2. The inputs barely moved")
c1, c2 = st.columns([3, 2])
with c1:
    st.altair_chart(psi_chart())
with c2:
    st.markdown(
        f"Every feature's PSI is **below 0.02**, far under the usual 0.1 \"small shift\" line. "
        "The orange ticks show the **noise floor** (2014 test vs. 2014 training data: same year, no real "
        "change). Occupation and birthplace sit at their noise floor: even through the 2014–2016 oil-price "
        "crash, the **mix of jobs** did not change measurably; pay did.\n\n"
        "**Education is the one clear drift:** more Texans hold higher degrees every year "
        "(KS 10× its noise floor)."
    )
with st.expander("KS statistic for the ordered features"):
    ks = ks_table()
    st.dataframe(ks.style.format("{:.4f}").format({f"{LAST_YEAR} / noise": "{:.1f}×"}))
    st.caption(
        "KS = largest gap between the two cumulative distributions (0 = identical). With ~130,000 people "
        "per year almost any difference is statistically significant, so KS is read as a size, not a test."
    )

# 3 ------------------------------------------------------------ SHAP over time
st.header("3. The models use the inputs the same way every year")
st.altair_chart(shap_chart())
st.markdown(
    "Feature importance (mean |SHAP|) is **flat** from 2014 to 2017 in both models: every change is "
    "0.05 or less, within the noise of a 1,000-person sample, and the ranking never changes. "
    "So the fairness drift does not come from the models reading different inputs. The two models do "
    "rely on different things: LR is dominated by occupation; XGBoost spreads its weight over age, "
    "education, hours and occupation. Both use sex and race/ethnicity directly."
)
st.caption("Unmitigated models only; 1,000 people per year. Compare years within a model, and models by ranking only.")

# 4 ------------------------------------------------------------ per-group TPR
st.header("4. So the old cutoff misses people unevenly")
choice = st.radio("Model", list(MODEL_CHOICES), horizontal=True)
wide = tpr_data(*MODEL_CHOICES[choice])
st.altair_chart(tpr_chart(wide))
lo, hi = wide.set_index("label")[str(LAST_YEAR)].idxmin(), wide.set_index("label")[str(LAST_YEAR)].idxmax()
st.markdown(
    "The newly high-earning people in groups the model already scored low (**women and Hispanic men**) "
    "mostly fall just below the 2014 cutoff, so those groups' true-positive rates **fall**: Hispanic men "
    "in every model, and women in every group large enough to measure. At the top, NH Asian men's rate "
    "**rises** in every model. The gap opens at both ends, and equalized odds difference (the largest "
    "TPR or FPR gap) goes up. Because the pattern is shared by all four models, it comes from the data, "
    "not from one model."
)
st.caption(
    f"Gray: {FIRST_YEAR} (test set). Red: {LAST_YEAR} (all people). Sorted by change, largest fall on top. "
    f"In {LAST_YEAR}, {choice}'s lowest TPR is {lo.rstrip(' *')} and its highest is {hi.rstrip(' *')}. "
    "* Small groups (about 40–60 people earning > $50K in the 2014 test set): ignore changes below ~0.05."
)

st.divider()
st.markdown(
    "**What this means for fixing it:** the ranking still works (AUC is flat) and the models still use "
    "the inputs the same way; the problem sits at the **decision cutoff**. That points to cutoff "
    "adjustments as a cheap fix, tested against retraining on the **Fix Comparison** page."
)

with st.expander("Caveats"):
    st.markdown(
        "- **Inflation share is likely a lower bound:** reported incomes cluster on round numbers, and "
        "many people report exactly $50,000, which counts as ≤ $50K under both lines.\n"
        "- **Noise floor for sex and race/ethnicity is zero by design:** the 2014 split was stratified on "
        "them, so judge their PSI by the raw value.\n"
        "- **The 2014 noise floor uses ~25,000 test rows vs. ~130,000 rows in later years,** so it slightly "
        "overstates noise (a conservative comparison).\n"
        "- National CPI-U is used for inflation, not a Texas-specific index."
    )