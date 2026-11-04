"""Model builders. Each pipeline = 2014-fitted preprocessing + classifier."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

from src.preprocess import build_preprocessor

from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient

import joblib
from src.data import ARTIFACTS_DIR

RANDOM_STATE = 42

MODEL_DIR = ARTIFACTS_DIR / "models"

# File name -> how the model is labelled in results (matches metrics_over_time.csv)
MODEL_INFO = {
    "lr":            {"model": "LR",      "mitigated": False},
    "lr_mitigated":  {"model": "LR",      "mitigated": True},
    "xgb":           {"model": "XGBoost", "mitigated": False},
    "xgb_mitigated": {"model": "XGBoost", "mitigated": True},
}

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

class MitigatedClassifier(ClassifierMixin, BaseEstimator):
    """ExponentiatedGradient (EqualizedOdds), stored compactly.

    After fitting, keeps only the inner models with non-zero weight and their
    weights (the full mitigator holds a copy of the training data and is dropped).
    predict_proba: weighted vote share of the inner models (same as Fairlearn's _pmf_predict).
    predict: 1 if vote share >= 0.5 (deterministic). Chosen over Fairlearn's randomized
    predict: on 2014 validation (LR) it kept most of the EOD reduction (0.598 -> 0.204)
    for 1.1 points of accuracy, vs. 0.118 for 3.9 points with randomized predictions.
    """

    def __init__(self, estimator=None, eps=0.01, max_iter=50):
        self.estimator = estimator
        self.eps = eps
        self.max_iter = max_iter

    def fit(self, X, y, sensitive_features):
        mitigator = ExponentiatedGradient(
            estimator=self.estimator,
            constraints=EqualizedOdds(),
            eps=self.eps,
            max_iter=self.max_iter,
        )
        mitigator.fit(X, y, sensitive_features=sensitive_features)
        self._keep_nonzero(mitigator)
        return self

    def _keep_nonzero(self, mitigator):
        """Store only the inner models with weight > 0, and their weights."""
        weights = np.asarray(mitigator.weights_, dtype=float)
        predictors = list(mitigator.predictors_)
        keep = weights > 0
        self.predictors_ = [p for p, k in zip(predictors, keep) if k]
        self.weights_ = weights[keep] / weights[keep].sum()
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p1 = sum(w * h.predict(X) for w, h in zip(self.weights_, self.predictors_))
        p1 = np.asarray(p1, dtype=float)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_mitigated(clf, X_train, y_train, sensitive_train):
    """Fit preprocessing + ExponentiatedGradient(EqualizedOdds) and return a Pipeline.

    Works around Pipeline not accepting sample_weight: fit the preprocessor first,
    mitigate on the transformed data, then assemble the two fitted pieces.
    sensitive_train: DataFrame of sensitive columns (e.g. race_eth and sex),
    aligned row for row with X_train and y_train.
    """
    pre = build_preprocessor().fit(X_train)
    Xt = pre.transform(X_train)
    model = MitigatedClassifier(estimator=clf).fit(
        Xt,
        np.asarray(y_train),
        sensitive_features=sensitive_train.reset_index(drop=True),
    )
    return Pipeline([("pre", pre), ("model", model)])

#load model
def load_models(model_dir=MODEL_DIR):
    """Load the four frozen 2014 models as {name: fitted Pipeline}.

    Names: lr, lr_mitigated, xgb, xgb_mitigated (see MODEL_INFO for labels).
    Used by Phase 2, Phase 4 and the app, so every part of the project
    evaluates exactly the same saved models. Never refits.
    """
    return {name: joblib.load(model_dir / f"{name}.joblib") for name in MODEL_INFO}