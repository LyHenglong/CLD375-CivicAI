"""Model comparison code (Phase 5 and 5b).

notebooks/03_model_comparison.ipynb calls these functions and explains the results.
Every model is the clean/encode pipeline from src/features.py plus a classifier.

Re-check the validation table from the project root (uses reports/best_settings.json):
    python -m src.train
"""

import json
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import features as ft

SEED = 42
CV_FOLDS = 5
BEST_SETTINGS_FILE = Path("reports/best_settings.json")
TUNING_FILE = Path("reports/tuning_results.csv")


def load_xy(split, folder=ft.PROCESSED_DIR):
    """Read one saved split and return (inputs, target)."""
    table = pd.read_parquet(Path(folder) / f"{split}.parquet")
    return table[ft.RAW_INPUT_COLUMNS], table[ft.TARGET]


# ---------------------------------------------------------------- baselines
class ComplaintTypeBaseline(ClassifierMixin, BaseEstimator):
    """Predicts from the complaint type alone.

    fit() stores, for each complaint type, the share of training requests that
    were resolved within 7 days. The prediction is that type's majority class.
    A type that was not in the training rows gets the overall share.
    """

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.classes_ = np.array([0, 1])
        self.overall_rate_ = float(y.mean())
        self.type_rates_ = y.groupby(X["complaint_type"]).mean()
        return self

    def predict_proba(self, X):
        rate = X["complaint_type"].map(self.type_rates_).fillna(self.overall_rate_).to_numpy(dtype=float)
        return np.column_stack([1 - rate, rate])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def make_baselines():
    return {
        # Always answers with the most common class of the training rows
        "Baseline: majority class": DummyClassifier(strategy="prior"),
        # Cleans the complaint type the same way as the models, then looks it up
        "Baseline: complaint type only": Pipeline([
            ("clean", ft.RequestCleaner(features=["complaint_type"])),
            ("model", ComplaintTypeBaseline())]),
    }


# ---------------------------------------------------------------- models
# The six candidates. All of them accept the sparse input of the pipeline.
CANDIDATES = ["Logistic Regression", "Decision Tree", "Random Forest",
              "Extra Trees", "XGBoost", "Bernoulli Naive Bayes"]
# Tuned in Phase 5 (results kept as they are) and added in Phase 5b
PHASE5_MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]
PHASE5B_MODELS = ["Decision Tree", "Extra Trees", "Bernoulli Naive Bayes"]


def class_ratio(y):
    """Rows of class 0 divided by rows of class 1 (about 0.21 here)."""
    return float((y == 0).sum() / (y == 1).sum())


def default_settings(y_train):
    """Reasonable first settings for each candidate (no tuning yet).

    Trees are kept shallow and small on purpose: the model has to run on a small
    EC2 server, and an unlimited tree would just memorise the training rows.
    """
    return {
        "Logistic Regression": {"C": 1.0},
        "Decision Tree": {"max_depth": 12, "min_samples_leaf": 20},
        "Random Forest": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5},
        "Extra Trees": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 5},
        "XGBoost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6,
                    "scale_pos_weight": round(class_ratio(y_train), 4)},
        "Bernoulli Naive Bayes": {"alpha": 1.0, "class_prior": [0.5, 0.5]},
    }


def make_model(name, settings):
    """Build the full pipeline (clean -> encode -> classifier) for one model.

    Every model handles the 17% minority class (class 0) in its own way:
    class weights, scale_pos_weight, or the class prior.
    """
    if name == "Logistic Regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED, **settings)
    elif name == "Decision Tree":
        classifier = DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **settings)
    elif name == "Random Forest":
        # balanced_subsample: class weights worked out inside each tree's sample
        classifier = RandomForestClassifier(class_weight="balanced_subsample", max_features="sqrt",
                                            random_state=SEED, n_jobs=-1, **settings)
    elif name == "Extra Trees":
        # Extra Trees does not resample rows (no bootstrap), so "balanced_subsample"
        # gives the same weights as "balanced" here: one set, from the whole training data
        classifier = ExtraTreesClassifier(class_weight="balanced_subsample", max_features="sqrt",
                                          random_state=SEED, n_jobs=-1, **settings)
    elif name == "XGBoost":
        # scale_pos_weight (in the settings) below 1 makes the common class 1 count less
        classifier = XGBClassifier(tree_method="hist", random_state=SEED, n_jobs=-1, **settings)
    elif name == "Bernoulli Naive Bayes":
        # It has no class weights. class_prior (in the settings) tells it how common to
        # assume each class is: [0.5, 0.5] = equally common, None = the real 17/83 shares.
        # It expects yes/no inputs, so it gets the encoder where every column is 0 or 1.
        return ft.build_pipeline(BernoulliNB(**settings), binary=True)
    else:
        raise ValueError(f"unknown model: {name}")
    return ft.build_pipeline(classifier)


def settings_to_try(y_train):
    """The small list of settings tried for each model.

    Logistic Regression, Random Forest and XGBoost: 6 to 8 settings, tuned in Phase 5.
    These lists must stay as they are, because the saved results belong to them.
    Decision Tree, Extra Trees and Naive Bayes (Phase 5b): 4 settings each, the
    first one being the default from default_settings().
    """
    first = default_settings(y_train)
    ratio = class_ratio(y_train)                       # fully balances the two classes
    half = round(float(np.sqrt(ratio)), 4)             # a milder, half-way weighting
    ratio = round(ratio, 4)
    return {
        "Logistic Regression": [{"C": c} for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]],
        "Decision Tree": [first["Decision Tree"],
                          {"max_depth": 20, "min_samples_leaf": 20},
                          {"max_depth": 30, "min_samples_leaf": 20},
                          {"max_depth": 40, "min_samples_leaf": 20}],
        "Random Forest": [      # 150 trees: the settings that were run in Phase 5
            {"n_estimators": 150, "max_depth": 20, "min_samples_leaf": 5},
            {"n_estimators": 150, "max_depth": 30, "min_samples_leaf": 5},
            {"n_estimators": 150, "max_depth": 40, "min_samples_leaf": 5},
            {"n_estimators": 150, "max_depth": 30, "min_samples_leaf": 10},
            {"n_estimators": 150, "max_depth": 40, "min_samples_leaf": 10},
            {"n_estimators": 150, "max_depth": 40, "min_samples_leaf": 20},
        ],
        "Extra Trees": [first["Extra Trees"],      # 100 trees keeps the saved file small
                        {"n_estimators": 100, "max_depth": 30, "min_samples_leaf": 5},
                        {"n_estimators": 100, "max_depth": 40, "min_samples_leaf": 5},
                        {"n_estimators": 100, "max_depth": 40, "min_samples_leaf": 10}],
        "XGBoost": [
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4, "scale_pos_weight": ratio},
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "scale_pos_weight": ratio},
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 8, "scale_pos_weight": ratio},
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 4, "scale_pos_weight": half},
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "scale_pos_weight": half},
            {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 8, "scale_pos_weight": half},
            {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 8, "scale_pos_weight": ratio},
            {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 8, "scale_pos_weight": half},
        ],
        "Bernoulli Naive Bayes": [first["Bernoulli Naive Bayes"],
                                  {"alpha": 1.0, "class_prior": None},
                                  {"alpha": 0.1, "class_prior": [0.5, 0.5]},
                                  {"alpha": 0.1, "class_prior": None}],
    }


# ---------------------------------------------------------------- cross-validation
def time_folds():
    """5 folds in time order: each trains on the past and validates on the next block."""
    return TimeSeriesSplit(n_splits=CV_FOLDS)


def cross_validate_model(model, X, y):
    """Mean and standard deviation of macro F1 and ROC-AUC over the time folds."""
    scores = cross_validate(model, X, y, cv=time_folds(), scoring={"macro_f1": "f1_macro", "roc_auc": "roc_auc"})
    return {"cv_macro_f1_mean": scores["test_macro_f1"].mean(), "cv_macro_f1_std": scores["test_macro_f1"].std(),
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(), "cv_roc_auc_std": scores["test_roc_auc"].std()}


def tune(name, settings_list, X, y):
    """Cross-validate every setting. Returns a table, best setting (highest mean macro F1) first."""
    rows = []
    for settings in settings_list:
        start = time.time()
        row = {"model": name, "settings": json.dumps(settings)}
        row.update(cross_validate_model(make_model(name, settings), X, y))
        row["cv_seconds"] = time.time() - start
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["cv_macro_f1_mean", "cv_roc_auc_mean"], ascending=False)
    return table.reset_index(drop=True)


def load_or_tune(name, settings_list, X, y, path=TUNING_FILE):
    """Return the saved tuning table for this model, or tune it now and save it.

    Results already in the file are kept exactly as they are (no re-tuning).
    Returns (table, True if it was loaded from the file).
    """
    path = Path(path)
    saved = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model", "settings"])
    wanted = {json.dumps(s) for s in settings_list}
    if set(saved.loc[saved["model"] == name, "settings"]) == wanted:
        return saved[saved["model"] == name].reset_index(drop=True), True
    table = tune(name, settings_list, X, y)
    pd.concat([saved[saved["model"] != name], table], ignore_index=True).to_csv(path, index=False)
    return table, False


# ---------------------------------------------------------------- evaluation
def evaluate(model, X, y):
    """All validation metrics for one fitted model. Returns (metrics, probabilities)."""
    predicted = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y, predicted),
        "precision_0": precision[0], "recall_0": recall[0], "f1_0": f1[0],     # class 0 = NOT resolved within 7 days
        "precision_1": precision[1], "recall_1": recall[1], "f1_1": f1[1],     # class 1 = resolved within 7 days
        "macro_f1": f1_score(y, predicted, average="macro"),
        "roc_auc": roc_auc_score(y, probability),
        # confusion matrix: "actual 0, predicted 0" and so on
        "actual0_pred0": int(tn), "actual0_pred1": int(fp), "actual1_pred0": int(fn), "actual1_pred1": int(tp),
    }
    return metrics, probability


def fit_timed(model, X, y):
    """Fit and return the training time in seconds."""
    start = time.time()
    model.fit(X, y)
    return time.time() - start


def saved_size_mb(model):
    """Size of the model when saved with joblib, in MB."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.joblib"
        joblib.dump(model, path)
        return path.stat().st_size / 1e6


def predict_one_ms(model, one_request, repeats=20):
    """Median time to predict a single request, in milliseconds."""
    model.predict_proba(one_request)                 # first call is slower (warm-up), not counted
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_proba(one_request)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def importance_by_feature(model):
    """Add the importance of the encoded columns back up to the original features."""
    names = model["encode"].get_feature_names_out()
    importance = pd.Series(model["model"].feature_importances_, index=names)
    features = sorted(model["clean"].features, key=len, reverse=True)

    def owner(encoded_name):
        rest = encoded_name.split("__", 1)[1].replace("missingindicator_", "")
        return next(f for f in features if rest.startswith(f))

    return importance.groupby(owner).sum().sort_values(ascending=False)


# ---------------------------------------------------------------- re-check
if __name__ == "__main__":
    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("val")                    # the test split is not used here
    best_settings = json.loads(BEST_SETTINGS_FILE.read_text(encoding="utf-8"))
    pipelines = make_baselines()
    pipelines.update({name: make_model(name, best_settings[name]) for name in CANDIDATES})
    print(f"{'model':32s} accuracy  prec_0  recall_0  f1_0    macro_f1  roc_auc  seconds")
    for model_name, pipeline in pipelines.items():
        seconds = fit_timed(pipeline, X_train, y_train)
        r, _ = evaluate(pipeline, X_val, y_val)
        print(f"{model_name:32s} {r['accuracy']:.4f}    {r['precision_0']:.4f}  {r['recall_0']:.4f}    "
              f"{r['f1_0']:.4f}  {r['macro_f1']:.4f}    {r['roc_auc']:.4f}   {seconds:6.0f}")
