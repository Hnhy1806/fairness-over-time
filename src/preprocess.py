from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column groups (shared by notebooks, models, and the app)
NUMERIC_COLS = ["AGEP", "WKHP"]
ORDINAL_COLS = ["SCHL"]
NOMINAL_COLS = ["COW", "MAR", "OCCP", "POBP", "RELP", "SEX", "race_eth"]
FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS

def build_preprocessor(min_frequency=30):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_COLS),
            ('ord', StandardScaler(), ORDINAL_COLS),
            ('nom', OneHotEncoder(
                min_frequency=30,
                handle_unknown='infrequent_if_exist',
                sparse_output=False
            ), NOMINAL_COLS)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
