"""Model builders. Each pipeline = 2014-fitted preprocessing + classifier."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.preprocess import build_preprocessor

RANDOM_STATE = 42

def make_lr():
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

def make_xgb(**params):
    defaults = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    defaults.update(params)
    return XGBClassifier(**defaults)

def make_pipeline_for(model):
    return Pipeline([
        ('pre', build_preprocessor()),
        ('model', model)
    ])