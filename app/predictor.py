"""Loads the saved model once and turns a validated request into a prediction.

Settings come from environment variables (all optional):
    CIVICAI_MODEL_PATH   the model file             default: models/model.joblib (in the project folder)
    CIVICAI_META_PATH    the model description      default: model_meta.json in the same folder as the model
    CIVICAI_LOG_DIR      where predictions.jsonl goes   default: logs/ in the project folder
    CIVICAI_WARMUP       "0" turns off the warm-up prediction at startup   default: on

The saved pipeline needs the "src" package (src/features.py, src/preprocessing.py) on the import
path. Start the server from the project root; the line below also adds the root, so it works
from another folder too.
"""

import json
import logging
import logging.handlers
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from app.schemas import PredictRequest

log = logging.getLogger("civicai")

LABELS = {1: "resolved within 7 days", 0: "not resolved within 7 days"}

# Fields whose text is turned into a category by the model. For each one the response says
# so when the value is not one the model knows.
CATEGORY_FIELDS = ["complaint_type", "descriptor", "location_type", "borough", "open_data_channel_type", "incident_zip"]

# Fields of the request that are sent to the model. Missing optional fields are None.
MODEL_FIELDS = ["complaint_type", "descriptor", "location_type", "borough", "incident_zip",
                "open_data_channel_type", "latitude", "longitude", "created_at"]

# Keys of model_meta.json that are not shown by GET /model-info (file names, deployment notes, version history)
HIDDEN_META_KEYS = {"file", "requires_on_server", "start_from", "previous_versions", "previous_version"}


class ModelNotAvailable(Exception):
    """The model (or its description file) could not be loaded."""


def _path_from_env(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else Path(default)


def installed_versions():
    """The versions of the libraries that matter for the saved model."""
    return {"python": platform.python_version(), **{name: metadata.version(name) for name in
            ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]}}


def check_versions(expected, installed=None):
    """Compare the installed libraries with the ones the model was trained with.

    Returns a list of plain-language mismatches (empty = everything matches).
    A different Python PATCH version (3.10.10 against 3.10.12) is harmless and is not reported.
    """
    installed = installed or installed_versions()
    problems = []
    for name, wanted in expected.items():
        have = installed.get(name)
        if have is None or have == wanted:
            continue
        if name == "python" and have.split(".")[:2] == wanted.split(".")[:2]:
            continue
        problems.append(f"{name} {have} is installed but the model was trained with {wanted}")
    return problems


def setup_prediction_log(directory):
    """A rotating JSON-lines file with one line per prediction. Returns the logger, or None."""
    logger = logging.getLogger("civicai.predictions")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):               # a new app (tests) must not keep writing to an old file
        logger.removeHandler(handler)
        handler.close()
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(directory / "predictions.jsonl", maxBytes=5_000_000,
                                                       backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        return logger
    except OSError as error:                            # a log problem must never stop the API
        log.warning("Prediction log disabled: cannot write to %s (%s)", directory, type(error).__name__)
        return None


class Predictor:
    """Holds the model, loaded once at startup, and answers requests."""

    def __init__(self):
        self.model = None
        self.meta = None
        self.version_mismatches = []
        self.prediction_log = None
        self.load_error = None

    @property
    def ready(self):
        return self.model is not None and self.meta is not None

    # ---------------------------------------------------------------- startup
    def load(self):
        model_path = _path_from_env("CIVICAI_MODEL_PATH", ROOT / "models" / "model.joblib")
        meta_path = _path_from_env("CIVICAI_META_PATH", model_path.parent / "model_meta.json")
        self.prediction_log = setup_prediction_log(_path_from_env("CIVICAI_LOG_DIR", ROOT / "logs"))
        try:
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.model = joblib.load(model_path)         # once, here; never inside a request
        except Exception as error:                       # noqa: BLE001 - the API must still start and answer 503
            self.model, self.meta = None, None
            self.load_error = type(error).__name__
            log.error("The model could not be loaded from %s (%s: %s). /health and /predict will answer 503.",
                      model_path, type(error).__name__, error)
            return self
        self.after_cleaning = self.model[1:]              # every step after the shared cleaning step
        self._use_one_thread_for_the_forest()
        self.version_mismatches = check_versions(self.meta["versions"])
        for problem in self.version_mismatches:
            log.warning("VERSION MISMATCH: %s. Predictions may differ from the evaluated model.", problem)
        if not self.version_mismatches:
            log.info("Installed Python and library versions match models/model_meta.json.")
        log.info("Model %s loaded (%s).", self.meta["model_version"], model_path.name)
        if os.environ.get("CIVICAI_WARMUP", "1") != "0":
            self._warm_up()
        return self

    def _use_one_thread_for_the_forest(self):
        """Only for a stacking model that contains a Random Forest (versions 1.0 and 2.0): the forest was
        fitted with n_jobs=-1, and for one request starting threads costs more than it saves, so it is set
        to one thread. The answers do not change. The deployed XGBoost model has no forest: nothing to do."""
        from src.features import get_stack
        try:
            forest = get_stack(self.model).named_estimators_["random_forest"]["model"]
            forest.n_jobs = 1
        except (KeyError, AttributeError):
            pass

    def _warm_up(self):
        """One throw-away prediction, so that the first real request is not slower than the rest."""
        start = time.perf_counter()
        sample = PredictRequest(complaint_type="Illegal Parking", borough="QUEENS", open_data_channel_type="ONLINE",
                                created_at="2026-01-05T09:30:00")
        self.predict(sample, log_it=False)
        log.info("Warm-up prediction took %.0f ms.", (time.perf_counter() - start) * 1000)

    # ---------------------------------------------------------------- information
    def public_meta(self):
        """models/model_meta.json without local paths and deployment notes."""
        if not self.ready:
            raise ModelNotAvailable
        return {key: value for key, value in self.meta.items() if key not in HIDDEN_META_KEYS}

    # ---------------------------------------------------------------- prediction
    def predict(self, request: PredictRequest, log_it=True):
        if not self.ready:
            raise ModelNotAvailable
        start = time.perf_counter()
        row = pd.DataFrame([{name: getattr(request, name) for name in MODEL_FIELDS}])

        # Pipeline.predict() is "clean the row, then ask the rest of the pipeline". Doing the two parts by hand
        # lets the cleaning run once, and the cleaned row is also used for the warnings.
        # The class comes from the model's own predict(), which applies its cut-off.
        cleaned = self.model["clean"].transform(row)
        probability = float(self.after_cleaning.predict_proba(cleaned)[0, 1])
        predicted_class = int(self.after_cleaning.predict(cleaned)[0])

        result = {
            "predicted_class": predicted_class,
            "label": LABELS[predicted_class],
            "probability_resolved_within_7d": round(probability, 6),
            "probability_not_resolved_within_7d": round(1 - probability, 6),
            "decision_threshold": float(self.meta["label"]["decision_threshold"]),
            "threshold_days": int(self.meta["label"]["threshold_days"]),
            "model_version": str(self.meta["model_version"]),
            "warnings": self._warnings(request, cleaned),
        }
        if log_it:
            self._log(request, result, (time.perf_counter() - start) * 1000)
        return result

    def _warnings(self, request, cleaned):
        """Plain-language notes about values the model could not use as given."""
        notes = []
        for field in CATEGORY_FIELDS:
            given = getattr(request, field)
            if given is None:
                continue
            used = cleaned.iloc[0][field]
            if used == "Other" and given.strip().upper() != "OTHER":
                extra = " (the city sometimes renames complaint types)" if field == "complaint_type" else ""
                notes.append(f"{field} '{given}' was not seen often enough in the training data{extra}, so the model "
                             f"treats it as 'Other'. The prediction relies on the other fields and may be less reliable.")
            elif used == "Unknown":
                if field == "incident_zip":
                    notes.append(f"incident_zip '{given}' is not a New York City ZIP code, so it is ignored.")
                else:
                    notes.append(f"{field} '{given}' is not a recognised value, so the model treats it as unknown.")
        if request.location_type is None:
            notes.append("location_type was not given. It is the model's most important input (it also tells the model which city agency "
                         "handles the request): in validation the model caught only about 28% of the slow requests without it, "
                         "against 82% with it. Send it if you can, written the way the 311 system records it.")
        if request.latitude is not None and pd.isna(cleaned.iloc[0]["latitude"]):
            from src.preprocessing import NYC_LAT_RANGE, NYC_LON_RANGE
            notes.append("The coordinates are outside the area the model was trained on "
                         f"(latitude {NYC_LAT_RANGE[0]} to {NYC_LAT_RANGE[1]}, longitude {NYC_LON_RANGE[0]} to {NYC_LON_RANGE[1]}), "
                         "so they are ignored.")
        return notes

    def _log(self, request, result, milliseconds):
        """One line per prediction. No coordinates, ZIP code, descriptor or place type are written,
        so the log cannot identify a place precisely."""
        if self.prediction_log is None:
            return
        self.prediction_log.info(json.dumps({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "complaint_type": request.complaint_type, "borough": request.borough,
            "channel": request.open_data_channel_type, "has_coordinates": request.latitude is not None,
            "predicted_class": result["predicted_class"], "probability": result["probability_resolved_within_7d"],
            "warnings": len(result["warnings"]), "milliseconds": round(milliseconds, 1),
        }))
