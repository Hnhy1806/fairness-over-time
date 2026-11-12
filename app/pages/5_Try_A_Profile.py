"""Try a Profile (Version A): one person -> 4 frozen 2014 models.

Shows each model's score and decision, a SHAP explanation for the two unmitigated
models, and a what-if grid that changes only sex and race/ethnicity.
The app never trains: models, background sample and lookups are read from artifacts.
"""
import altair as alt
import pandas as pd
import streamlit as st

from src.explain import explain
from src.models import MODEL_INFO
from utils import get_lookups, get_models, load_background

st.set_page_config(page_title="Try a Profile", layout="wide")

# ---------------------------------------------------------------- constants
X_COLS = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "race_eth"]  # same order as X
FEATURE_NAMES = {
    "AGEP": "Age", "COW": "Class of worker", "SCHL": "Education", "MAR": "Marital status",
    "OCCP": "Occupation", "POBP": "Place of birth", "RELP": "Relationship in household",
    "WKHP": "Hours worked per week", "SEX": "Sex", "race_eth": "Race/ethnicity",
}
SEX_LABELS = {1.0: "Male", 2.0: "Female"}
RACE_GROUPS = ["Hispanic", "NH White", "NH Black", "NH Asian", "NH Other"]
RACE_HELP = (
    "Hispanic = any race; NH = non-Hispanic. NH Other = American Indian/Alaska Native, "
    "Pacific Islander, some other race, or two or more races."
)
THRESHOLD = 0.5
EXPLAINED = ["lr", "xgb"]  # SHAP only for the unmitigated models (see caption)


# ---------------------------------------------------------------- data
models = get_models()
background = load_background()
lookups = get_lookups()


def options(feature, model_codes_only=True):
    """{code: label} for one coded feature, optionally only codes the model kept a column for."""
    rows = lookups[lookups["feature"] == feature]
    if model_codes_only:
        rows = rows[rows["in_model"]]
    return dict(zip(rows["code"], rows["label"]))


def default(feature):
    """Starting value: median (age, hours) or most common code in the 2014 background sample."""
    col = background[feature]
    return float(col.median()) if feature in ("AGEP", "WKHP") else col.mode().iloc[0]


@st.cache_data(show_spinner=False)
def scores(person):
    """Score (P(> $50K), or vote share for mitigated) and decision from all 4 models."""
    rows = []
    for name, pipe in models.items():
        score = float(pipe.predict_proba(person)[:, 1][0])
        rows.append({**MODEL_INFO[name], "score": score, "decision": score >= THRESHOLD})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def shap_values(person):
    """SHAP contributions (log-odds) per raw feature for the unmitigated LR and XGBoost."""
    out = []
    for name in EXPLAINED:
        values = explain(models[name], person, background).iloc[0]
        out.append(pd.DataFrame({
            "model": MODEL_INFO[name]["model"],
            "feature": values.index,
            "shap": values.astype(float).values,
        }))
    return pd.concat(out, ignore_index=True)


@st.cache_data(show_spinner=False)
def what_if(person):
    """Scores of all 4 models for the same person under every sex × race/ethnicity combination."""
    rows = []
    for sex_code, sex_label in SEX_LABELS.items():
        for race_group in RACE_GROUPS:
            variant = person.assign(SEX=sex_code, race_eth=race_group)
            row = {"Sex": sex_label, "Race/ethnicity": race_group}
            for name, pipe in models.items():
                info = MODEL_INFO[name]
                column = f"{info['model']} {'mitigated' if info['mitigated'] else 'unmitigated'}"
                row[column] = float(pipe.predict_proba(variant)[:, 1][0])
            rows.append(row)
    return pd.DataFrame(rows)


def value_label(feature, value):
    """Readable feature = value text for the SHAP chart, e.g. 'Occupation = Registered Nurses'."""
    if feature in ("AGEP", "WKHP"):
        return f"{FEATURE_NAMES[feature]} = {value:g}"
    if feature == "SEX":
        return f"Sex = {SEX_LABELS[value]}"
    if feature == "race_eth":
        return f"Race/ethnicity = {value}"
    label = options(feature, model_codes_only=False).get(value, f"code {value:g}")
    return f"{FEATURE_NAMES[feature]} = {label}"


def shap_chart(df):
    """Horizontal bars per feature, one panel per model, largest contribution on top."""
    df = df.assign(direction=df["shap"].map(lambda v: "Toward > $50K" if v > 0 else "Toward ≤ $50K"))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("shap:Q", title="Contribution to the score (log-odds)"),
            y=alt.Y("label:N", title=None, sort=alt.EncodingSortField("abs_shap", order="descending"),
                    axis=alt.Axis(labelLimit=320)),
            color=alt.Color(
                "direction:N", title=None,
                scale=alt.Scale(domain=["Toward > $50K", "Toward ≤ $50K"], range=["#2ca02c", "#d62728"]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("shap:Q", title="SHAP (log-odds)", format="+.3f"),
            ],
        )
        .properties(width=300, height=300)
        .facet(column=alt.Column("model:N", title=None, sort=["LR", "XGBoost"]))
        .resolve_scale(y="independent")
    )


def select(label, feature, **kwargs):
    """Selectbox over the model's codes for one feature, showing labels, starting at the default."""
    opts = options(feature)
    codes = list(opts)
    return st.selectbox(label, codes, format_func=opts.get, index=codes.index(default(feature)), **kwargs)


# ---------------------------------------------------------------- page: inputs
st.title("Try a Profile")
st.markdown(
    "Describe one person and see what the **four frozen 2014 models** predict: "
    "does this person earn **more than $50K** a year? The models were trained on Texas 2014 data "
    "and never retrained. The form starts with the most common values in the 2014 data."
)

with st.form("profile"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 17, 95, int(default("AGEP")))
        hours = st.slider("Hours worked per week", 1, 99, int(default("WKHP")))
        sex = st.radio("Sex", list(SEX_LABELS), format_func=SEX_LABELS.get,
                       index=list(SEX_LABELS).index(default("SEX")), horizontal=True)
        race = st.selectbox("Race/ethnicity", RACE_GROUPS,
                            index=RACE_GROUPS.index(default("race_eth")), help=RACE_HELP)
    with c2:
        schl = select("Education", "SCHL")
        occp = select("Occupation (type to search)", "OCCP",
                      help=f"The {len(options('OCCP'))} occupations with at least 30 people in the 2014 "
                           "training data; the model groups rarer ones together.")
        cow = select("Class of worker", "COW")
    with c3:
        mar = select("Marital status", "MAR")
        relp = select("Relationship in household", "RELP",
                      help="Relationship to the household's reference person (the person who filled in the survey).")
        pobp = select("Place of birth (type to search)", "POBP",
                      help=f"The {len(options('POBP'))} birthplaces with at least 30 people in the 2014 training data.")
    st.form_submit_button("Predict", type="primary")

person = pd.DataFrame([{
    "AGEP": float(age), "COW": cow, "SCHL": schl, "MAR": mar, "OCCP": occp, "POBP": pobp,
    "RELP": relp, "WKHP": float(hours), "SEX": sex, "race_eth": race,
}])[X_COLS]

# ---------------------------------------------------------------- page: predictions
st.subheader("Predictions")
pred = scores(person)
for col, (_, r) in zip(st.columns(4), pred.iterrows()):
    version = "mitigated" if r["mitigated"] else "unmitigated"
    col.metric(
        label=f"{r['model']} · {version}",
        value="> $50K" if r["decision"] else "≤ $50K",
        delta=f"score {r['score']:.2f}",
        delta_color="off",
    )
st.caption(
    f"Decision = score ≥ {THRESHOLD}. Unmitigated score = predicted probability of earning > $50K. "
    "Mitigated score = share of the fairness-constrained ensemble voting '> $50K' (not a calibrated probability)."
)

# ---------------------------------------------------------------- page: SHAP
st.subheader("Why? Feature contributions (SHAP)")
shap_df = shap_values(person)
shap_df["label"] = [value_label(f, person.iloc[0][f]) for f in shap_df["feature"]]
shap_df["abs_shap"] = shap_df["shap"].abs()
st.altair_chart(shap_chart(shap_df))
st.caption(
    "Each bar shows how much one feature moved this person's score compared with an average person "
    "from a 2014 reference sample (log-odds; exact for LR, TreeSHAP for XGBoost). "
    "The starting point is that average person's score, and about two-thirds of people earn ≤ $50K, "
    "so a person with mostly green bars can still end up below the cutoff. "
    "Shown for the unmitigated models only: the mitigated models are weighted votes of many inner "
    "models, which SHAP does not explain cleanly. Compare features within one model; the two models "
    "use different reference points, so compare them by ranking only."
)

# ---------------------------------------------------------------- page: what-if
st.subheader("What if only sex or race/ethnicity changed?")
grid = what_if(person)
score_cols = [c for c in grid.columns if c not in ("Sex", "Race/ethnicity")]


def highlight_actual(row):
    """Bold the row that matches the person as entered."""
    actual = row["Sex"] == SEX_LABELS[sex] and row["Race/ethnicity"] == race
    return ["font-weight: bold; border: 2px solid orange" if actual else ""] * len(row)


st.dataframe(
    grid.style.apply(highlight_actual, axis=1).format("{:.2f}", subset=score_cols),
    hide_index=True,
)
spread = grid[score_cols].max() - grid[score_cols].min()
st.markdown(
    "**Largest score difference across the 10 combinations:** "
    + " · ".join(f"{c}: {spread[c]:.2f}" for c in score_cols)
)
st.caption(
    "Bold row = the person as entered. Everything else (occupation, hours, household role…) is held "
    "fixed, so this shows only the models' **direct** use of sex and race/ethnicity. It does not capture "
    "indirect effects through proxy features such as occupation or household role, which also differ "
    "between groups in the real data."
)

with st.expander("About this page"):
    st.markdown(
        "- The four models are the frozen 2014 models used throughout this app; nothing is retrained here.\n"
        "- Sex and race/ethnicity are model inputs on purpose, so that SHAP and the what-if grid can measure "
        "how much the models rely on them. This is a research model for studying fairness over time, not a "
        "deployed decision system (using protected attributes directly is generally not allowed in US "
        "hiring or lending decisions).\n"
        "- Occupation and birthplace labels come from the Census 2017 ACS PUMS data dictionary; the codes are "
        "the same in 2014–2017."
    )