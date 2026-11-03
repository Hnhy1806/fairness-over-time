from pathlib import Path
from folktables import ACSDataSource, ACSIncome, acs
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


#race/ethnicity function
def make_race_eth(df):
    """Combine RAC1P and HISP into 5 race/ethnicity groups.

    HISP == 1 means NOT Hispanic; any other HISP code is a Hispanic origin.
    Works on a DataFrame of any length (1 row for the app, many for analysis).
    """
    conditions = [
        df['HISP'] != 1, # TODO: Hispanic condition
        df['RAC1P'] == 1, # TODO: NH White condition
        df['RAC1P'] == 2, # TODO: NH Black condition
        df['RAC1P'] == 6, # TODO: NH Asian condition
    ]
    labels = ['Hispanic', 'NH White', 'NH Black', 'NH Asian']   # TODO: the four labels, in the same order
    return pd.Series(
        np.select(conditions, labels, default='NH Other'),   # TODO
        index=df.index,
        name="race_eth",
    )


#loading data
def load_acs(year, state="TX"):
    """Load ACSIncome features, label, and group for one year and state.
    
    Returns X (10 features, with race_eth replacing RAC1P), y (income > $50K),
   and meta (race_eth, SEX, income in dollars, ADJINC, PWGTP) for the same rows.
   """   
    source = ACSDataSource(
        survey_year=str(year),
        horizon="1-Year",
        survey="person",
        root_dir=str(DATA_DIR),
    )
    raw = source.get_data(states=[state], download=True)
    X, y, _ = ACSIncome.df_to_pandas(raw)
    
    filtered = acs.adult_filter(raw).reset_index(drop=True)
    assert (filtered[X.columns].to_numpy() == X.to_numpy()).all(), "raw and X are not aligned"
    
    meta = pd.DataFrame({
        "race_eth": make_race_eth(filtered),    # TODO: your new function
        "SEX": filtered['SEX'],
        "income": filtered['PINCP'],      # dollar amount from the filtered data
        "ADJINC": filtered['ADJINC'],
        "PWGTP": filtered['PWGTP'],
    })

    X = X.drop(columns=['RAC1P'])
    X['race_eth'] = meta['race_eth']

    return X, y, meta


#split dataset
def split_data(y, meta, seed=42, min_stratum=10):
    """Split into 60/20/20 train/val/test, stratified by label x race x sex."""

    # 2a: build strata
    strata = y['PINCP'].astype(str) + "_" + meta['race_eth'].astype(str) + '_' + meta['SEX'].astype(str)
    # 2b: merge strata smaller than min_stratum into "rare"
    counts = strata.value_counts()
    rare = counts[counts<min_stratum].index
    strata = strata.mask(strata.isin(rare), 'rare')
    # 2c: the two splits, using seed
    train_idx, temp_idx = train_test_split(
    y.index,
    test_size=0.4,
    stratify=strata,
    random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,  # split remaining 40% into 20% val + 20% test
        stratify=strata.loc[temp_idx],
        random_state=seed
    )
    return train_idx, val_idx, test_idx


#save the split
def save_split(train_idx, val_idx, test_idx, path=ARTIFACTS_DIR/ "results"/ "split_2014.csv"):
    split_df = pd.concat([
        pd.DataFrame({'row_id': train_idx, 'split': 'train'}),
        pd.DataFrame({'row_id': val_idx, 'split': 'val'}),
        pd.DataFrame({'row_id': test_idx, 'split': 'test'})
    ]).sort_values('row_id')
    split_df.to_csv(path, index=False)
    return split_df


#load the split
def load_split(path=ARTIFACTS_DIR / "results" / "split_2014.csv"):
    """Load the saved split and return train, val, and test row ids."""
    split_df = pd.read_csv(path)
    train_idx = split_df.loc[split_df["split"] == "train", "row_id"].values
    val_idx = split_df.loc[split_df["split"] == "val", "row_id"].values
    test_idx = split_df.loc[split_df["split"] == "test", "row_id"].values
    return train_idx, val_idx, test_idx