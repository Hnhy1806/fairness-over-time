"""Code -> label lookups for the app's input widgets (Try a Profile).

Built once from the Census 2017 ACS PUMS data dictionary (CSV). The 2017 dictionary is
used for all years because the codes of these features are the same in 2014-2017
(OCCP checked in Phase 2; RELP and SCHL codes changed only in 2019), and 2017 is the
first year the Census publishes the dictionary as CSV.

Output: data/lookups/feature_codes.csv with columns
    feature, code, label, in_model
- code: float, the same type as the model inputs in X (e.g. 3255.0)
- in_model: True if the fitted 2014 encoder kept a column for this code; False means
  the model sees it only as part of the "infrequent" bucket (fewer than 30 training rows).
  SCHL is ordinal (scaled, not one-hot), so all its codes are True.

Build (once, from the repo root):  python -m src.lookups
"""
import pandas as pd

from src.data import DATA_DIR, LOOKUP_PATH

DICT_PATH = DATA_DIR / "PUMS_Data_Dictionary_2017.csv"  # data/raw/, not committed
LOOKUP_FEATURES = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP"]
DICT_COLS = ["kind", "variable", "type", "length", "code_min", "code_max", "label"]


def read_dictionary(dict_path=DICT_PATH, features=LOOKUP_FEATURES):
    """Single-code value rows (VAL) of the PUMS dictionary for the given features.

    Skips N/A codes written as blanks ("b", "bbbb") and value ranges (code_min != code_max).
    """
    raw = pd.read_csv(dict_path, header=None, names=DICT_COLS, dtype=str)
    vals = raw[(raw["kind"] == "VAL") & raw["variable"].isin(features)]
    vals = vals[(vals["code_min"] == vals["code_max"]) & vals["code_min"].str.isdigit()]
    return pd.DataFrame({
        "feature": vals["variable"],
        "code": vals["code_min"].astype(float),   # "0010" -> 10.0, same as X
        "label": vals["label"].str.strip(),
    }).reset_index(drop=True)


def kept_codes(pipe):
    """{feature: set of codes} that the fitted one-hot encoder keeps as their own column."""
    ohe = pipe.named_steps["pre"].named_transformers_["nom"]
    kept = {}
    for col, cats, infrequent in zip(ohe.feature_names_in_, ohe.categories_, ohe.infrequent_categories_):
        rare = set() if infrequent is None else set(infrequent)
        kept[col] = {c for c in cats if c not in rare}
    return kept


def all_model_codes(pipe):
    """{feature: set of every code the encoder saw in 2014 training, kept or infrequent}."""
    ohe = pipe.named_steps["pre"].named_transformers_["nom"]
    return {col: set(cats) for col, cats in zip(ohe.feature_names_in_, ohe.categories_)}


def build_lookups(pipe, dict_path=DICT_PATH):
    """Lookup table (feature, code, label, in_model) for LOOKUP_FEATURES.

    pipe: any frozen unmitigated pipeline (all 4 models share the same 2014 preprocessing).
    Codes seen in 2014 training but missing from the 2017 dictionary (codes the Census
    dropped or renumbered) get a placeholder label instead of a guessed one, and a warning.
    """
    lookups = read_dictionary(dict_path)

    seen = all_model_codes(pipe)
    extra = []
    for feature in LOOKUP_FEATURES:
        if feature not in seen:  # SCHL is ordinal, not in the encoder
            continue
        missing = seen[feature] - set(lookups.loc[lookups["feature"] == feature, "code"])
        for code in sorted(missing):
            print(f"Warning: {feature} code {code:g} is in the 2014 data but not in the dictionary")
            extra.append({"feature": feature, "code": float(code),
                          "label": f"Code {code:g} (not in the 2017 dictionary)"})
    if extra:
        lookups = pd.concat([lookups, pd.DataFrame(extra)], ignore_index=True)

    kept = kept_codes(pipe)
    lookups["in_model"] = [
        True if feature == "SCHL" else code in kept[feature]
        for feature, code in zip(lookups["feature"], lookups["code"])
    ]
    order = {f: i for i, f in enumerate(LOOKUP_FEATURES)}
    return lookups.sort_values(["feature", "code"], key=lambda s: s.map(order) if s.name == "feature" else s,
                               ignore_index=True)


def load_lookups(path=LOOKUP_PATH):
    """Read the saved lookup table (used by the app)."""
    return pd.read_csv(path)


if __name__ == "__main__":
    from src.models import load_models

    table = build_lookups(load_models()["lr"])
    LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(LOOKUP_PATH, index=False)
    summary = table.groupby("feature")["in_model"].agg(codes="size", in_model="sum")
    print(f"Saved {len(table)} rows to {LOOKUP_PATH}\n{summary}")
