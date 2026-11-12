"""Tests of the CivicAI API (Phase 7). Run from the project root:  python -m pytest

They start the real app in-process (FastAPI's TestClient), which loads the real
models/model.joblib, and send requests to it. The valid and invalid requests come from
data/sample_requests.json (validation-split rows; the test splits are never used here).
"""

import json
import logging
import re
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.predictor import check_versions
from app.schemas import PredictRequest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = json.loads((ROOT / "data" / "sample_requests.json").read_text(encoding="utf-8"))
META = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
VALID = SAMPLES["valid_requests"]
INVALID = SAMPLES["invalid_requests"]
BASE = {"complaint_type": "Noise - Residential", "borough": "BROOKLYN", "open_data_channel_type": "PHONE",
        "created_at": "2026-11-09T22:15:00"}


def fields_named(response):
    return [problem["field"] for problem in response.json()["errors"]]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    for name in ["CIVICAI_MODEL_PATH", "CIVICAI_META_PATH"]:
        patch.delenv(name, raising=False)
    patch.setenv("CIVICAI_LOG_DIR", str(tmp_path_factory.mktemp("logs")))
    with TestClient(create_app()) as test_client:
        yield test_client
    patch.undo()


@pytest.fixture(scope="module")
def saved_model():
    """The saved pipeline loaded directly, to compare with what the API says."""
    return joblib.load(ROOT / "models" / "model.joblib")


# ---------------------------------------------------------------- service endpoints
def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == META["model_version"]
    assert body["versions_match_model_meta"] is True and body["version_mismatches"] == []


def test_model_info_is_the_meta_file_without_paths(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    info = response.json()
    for key in ["model_version", "trained_on", "versions", "input_fields", "label", "split", "final_estimator"]:
        assert key in info
    assert info["label"]["decision_threshold"] == META["label"]["decision_threshold"]
    assert info["label"]["1"] == "resolved within 7 days" and info["label"]["0"] == "not resolved within 7 days"
    assert info["split"]["test"]["to"] == META["split"]["test"]["to"]
    for hidden in ["file", "requires_on_server", "start_from"]:
        assert hidden not in info
    text = json.dumps(info)
    assert ".joblib" not in text and not re.search(r"[A-Za-z]:\\", text) and "/home/" not in text


def test_docs_and_openapi_list_the_routes(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/model-info", "/predict"} <= set(paths)


# ---------------------------------------------------------------- valid requests
@pytest.mark.parametrize("sample", VALID, ids=[v["name"] for v in VALID])
def test_valid_sample_matches_the_saved_probability(client, sample):
    response = client.post("/predict", json=sample["request"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert abs(body["probability_resolved_within_7d"] - sample["expected_probability_within_threshold"]) < 1e-6
    assert body["predicted_class"] == sample["expected_predicted_class"]
    assert body["label"] == META["label"][str(body["predicted_class"])]
    assert abs(body["probability_resolved_within_7d"] + body["probability_not_resolved_within_7d"] - 1) < 2e-6
    assert body["decision_threshold"] == META["label"]["decision_threshold"] and body["threshold_days"] == 7
    assert body["model_version"] == META["model_version"]
    assert body["warnings"] == []


@pytest.mark.parametrize("sample", VALID, ids=[v["name"] for v in VALID])
def test_api_class_equals_model_predict(client, saved_model, sample):
    row = pd.DataFrame([sample["request"]])
    body = client.post("/predict", json=sample["request"]).json()
    assert body["predicted_class"] == int(saved_model.predict(row)[0])
    assert abs(body["probability_resolved_within_7d"] - float(saved_model.predict_proba(row)[0, 1])) < 1e-6
    cutoff = META["label"]["decision_threshold"]
    assert body["predicted_class"] == int(saved_model.predict_proba(row)[0, 1] >= cutoff)       # the cut-off is not 0.5


def test_cutoff_is_applied_not_0_5(client, saved_model):
    """A request whose probability lies between 0.5 and the cut-off must come out as class 0.

    "Maintenance or Facility" filed by phone in Brooklyn gets a probability of about 0.55 from
    this model, and the cut-off is 0.61. If the API compared the probability with 0.5 it would say 1.
    """
    cutoff = META["label"]["decision_threshold"]
    request = {"complaint_type": "Maintenance or Facility", "borough": "BROOKLYN", "open_data_channel_type": "PHONE",
               "created_at": "2026-02-10T14:00:00"}
    probability = float(saved_model.predict_proba(pd.DataFrame([request]))[0, 1])
    assert 0.5 <= probability < cutoff, f"the premise of this test no longer holds (probability {probability:.3f})"
    body = client.post("/predict", json=request).json()
    assert body["predicted_class"] == 0 and body["label"] == "not resolved within 7 days"
    assert 0.5 <= body["probability_resolved_within_7d"] < cutoff
    assert body["predicted_class"] == int(saved_model.predict(pd.DataFrame([request]))[0])


def test_optional_fields_can_be_left_out_or_null(client):
    assert client.post("/predict", json=BASE).status_code == 200
    nulls = {**BASE, "descriptor": None, "location_type": None, "incident_zip": None, "latitude": None, "longitude": None}
    assert client.post("/predict", json=nulls).status_code == 200


def test_case_spaces_and_time_zone_are_tolerated(client):
    messy = {**BASE, "borough": "  brooklyn ", "complaint_type": "noise - residential", "created_at": "2026-11-09T22:15:00Z"}
    body = client.post("/predict", json=messy).json()
    assert body["warnings"] == []


# ---------------------------------------------------------------- invalid requests
@pytest.mark.parametrize("sample", INVALID, ids=[v["name"] for v in INVALID])
def test_invalid_sample_returns_422_and_names_the_field(client, sample):
    response = client.post("/predict", json=sample["request"])
    assert response.status_code == sample["expected_status"] == 422
    assert sample["expected_error_field"] in fields_named(response)
    body = response.json()
    assert body["error"] == "validation_error" and sample["expected_error_field"] in body["message"]
    assert "Traceback" not in response.text


BAD_PAYLOADS = [
    ("missing complaint_type", {"complaint_type": None, "_drop": ["complaint_type"]}, "complaint_type"),
    ("missing open_data_channel_type", {"_drop": ["open_data_channel_type"]}, "open_data_channel_type"),
    ("missing created_at", {"_drop": ["created_at"]}, "created_at"),
    ("empty borough", {"borough": ""}, "borough"),
    ("blank complaint type", {"complaint_type": "   "}, "complaint_type"),
    ("empty descriptor", {"descriptor": ""}, "descriptor"),
    ("number as borough", {"borough": 5}, "borough"),
    ("number as zip", {"incident_zip": 11201}, "incident_zip"),
    ("zip with 4 digits", {"incident_zip": "1120"}, "incident_zip"),
    ("zip with letters", {"incident_zip": "11A01"}, "incident_zip"),
    ("latitude as text", {"latitude": "40.7", "longitude": -73.9}, "latitude"),
    ("longitude as boolean", {"latitude": 40.7, "longitude": True}, "longitude"),
    ("latitude impossible", {"latitude": 91.5, "longitude": -73.9}, "latitude"),
    ("longitude impossible", {"latitude": 40.7, "longitude": -200.0}, "longitude"),
    ("latitude without longitude", {"latitude": 40.7}, "longitude"),
    ("longitude without latitude", {"longitude": -73.9}, "latitude"),
    ("date as text", {"created_at": "yesterday"}, "created_at"),
    ("date without time", {"created_at": "2026-11-09"}, "created_at"),
    ("hour 25", {"created_at": "2026-11-09T25:00:00"}, "created_at"),
    ("month 13", {"created_at": "2026-13-09T10:00:00"}, "created_at"),
    ("date as number", {"created_at": 1700000000}, "created_at"),
    ("date in the future", {"created_at": "2031-01-01T10:00:00"}, "created_at"),
    ("date before 2020", {"created_at": "2019-12-31T10:00:00"}, "created_at"),
    ("unknown extra field", {"hour": 22}, "hour"),
    ("misspelt field", {"borogh": "BRONX"}, "borogh"),
    ("text far too long", {"complaint_type": "x" * 500}, "complaint_type"),
]


@pytest.mark.parametrize("label,change,field", BAD_PAYLOADS, ids=[b[0] for b in BAD_PAYLOADS])
def test_bad_input_returns_422_naming_the_field(client, label, change, field):
    payload = dict(BASE)
    for name in change.get("_drop", []):
        payload.pop(name)
    payload.update({k: v for k, v in change.items() if k != "_drop" and not (k == "complaint_type" and v is None)})
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, response.text
    assert field in fields_named(response)
    message = next(p["message"] for p in response.json()["errors"] if p["field"] == field)
    assert len(message) > 15 and "Traceback" not in response.text         # an understandable sentence


def test_broken_body_returns_422(client):
    broken = client.post("/predict", content="{not json", headers={"content-type": "application/json"})
    assert broken.status_code == 422 and "body" in fields_named(broken)
    listed = client.post("/predict", json=[BASE])
    assert listed.status_code == 422 and "body" in fields_named(listed)
    empty = client.post("/predict", json={})
    assert empty.status_code == 422
    assert {"complaint_type", "borough", "open_data_channel_type", "created_at"} <= set(fields_named(empty))


# ---------------------------------------------------------------- coordinates
@pytest.mark.parametrize("label,point,fields", [
    ("London", (51.5, -0.12), {"latitude", "longitude"}),
    ("Boston", (42.36, -71.06), {"latitude", "longitude"}),
    ("just north of the box", (41.2, -73.9), {"latitude"}),
    ("just east of the box", (40.7, -73.4), {"longitude"}),
    ("the Atlantic", (40.7, -75.0), {"longitude"}),
])
def test_points_outside_new_york_are_rejected(client, label, point, fields):
    response = client.post("/predict", json={**BASE, "latitude": point[0], "longitude": point[1]})
    assert response.status_code == 422
    assert fields <= set(fields_named(response))
    assert "New York" in response.text


def test_point_inside_the_generous_box_but_outside_the_training_area_gets_a_warning(client):
    response = client.post("/predict", json={**BASE, "latitude": 40.35, "longitude": -74.3})
    assert response.status_code == 200
    assert any("coordinates" in w and "ignored" in w for w in response.json()["warnings"])


# ---------------------------------------------------------------- unseen categories
def test_unseen_complaint_type_is_accepted_with_a_warning(client, saved_model):
    payload = {**BASE, "complaint_type": "Drone Delivery Complaint"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in (0, 1) and 0 <= body["probability_resolved_within_7d"] <= 1
    assert len(body["warnings"]) == 1
    assert "Drone Delivery Complaint" in body["warnings"][0] and "'Other'" in body["warnings"][0]
    assert body["predicted_class"] == int(saved_model.predict(pd.DataFrame([payload]))[0])


def test_other_unseen_values_also_warn(client):
    payload = {**BASE, "descriptor": "Never seen before text", "location_type": "Rooftop", "borough": "Unspecified",
               "incident_zip": "99999"}
    warnings = client.post("/predict", json=payload).json()["warnings"]
    for field in ["descriptor", "location_type", "borough", "incident_zip"]:
        assert any(w.startswith(field) or w.startswith(f"{field} ") or f"{field} '" in w for w in warnings), (field, warnings)


# ---------------------------------------------------------------- model missing -> 503
def test_missing_model_gives_503_with_a_clear_message(tmp_path, monkeypatch):
    missing = tmp_path / "no_such_model.joblib"
    monkeypatch.setenv("CIVICAI_MODEL_PATH", str(missing))
    monkeypatch.setenv("CIVICAI_META_PATH", str(ROOT / "models" / "model_meta.json"))
    monkeypatch.setenv("CIVICAI_LOG_DIR", str(tmp_path / "logs"))
    with TestClient(create_app()) as broken:
        health = broken.get("/health")
        predict = broken.post("/predict", json=BASE)
        info = broken.get("/model-info")
    assert health.status_code == 503 and health.json()["model_loaded"] is False
    assert predict.status_code == 503 and predict.json()["error"] == "model_unavailable"
    assert info.status_code == 503
    for response in (health, predict, info):
        assert "not loaded" in response.text and "Traceback" not in response.text
        assert "no_such_model" not in response.text and str(tmp_path) not in response.text      # no local paths


# ---------------------------------------------------------------- version check
def test_check_versions_reports_mismatches():
    installed = {"python": "3.10.12", "scikit-learn": "1.7.2", "numpy": "2.2.6"}
    assert check_versions({"python": "3.10.10", "scikit-learn": "1.7.2", "numpy": "2.2.6"}, installed) == []   # patch only
    assert check_versions({"python": "3.11.4"}, installed) == ["python 3.10.12 is installed but the model was trained with 3.11.4"]
    problems = check_versions({"scikit-learn": "1.6.0", "numpy": "2.2.6"}, installed)
    assert problems == ["scikit-learn 1.7.2 is installed but the model was trained with 1.6.0"]


def test_startup_logs_a_warning_when_versions_differ(tmp_path, monkeypatch, caplog):
    changed = json.loads(json.dumps(META))
    changed["versions"]["scikit-learn"] = "0.20.0"
    meta_file = tmp_path / "model_meta.json"
    meta_file.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setenv("CIVICAI_MODEL_PATH", str(ROOT / "models" / "model.joblib"))
    monkeypatch.setenv("CIVICAI_META_PATH", str(meta_file))
    monkeypatch.setenv("CIVICAI_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CIVICAI_WARMUP", "0")
    monkeypatch.setattr(logging.getLogger("civicai"), "propagate", True)       # so that caplog can see it
    with caplog.at_level(logging.WARNING, logger="civicai"):
        with TestClient(create_app()) as other:
            health = other.get("/health").json()
    assert any("VERSION MISMATCH" in r.message and "scikit-learn" in r.message for r in caplog.records)
    assert health["versions_match_model_meta"] is False and "scikit-learn" in health["version_mismatches"][0]


# ---------------------------------------------------------------- logging and consistency
def test_prediction_log_has_no_precise_location(tmp_path, monkeypatch):
    monkeypatch.setenv("CIVICAI_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("CIVICAI_MODEL_PATH", raising=False)
    monkeypatch.delenv("CIVICAI_META_PATH", raising=False)
    monkeypatch.setenv("CIVICAI_WARMUP", "0")
    request = {**BASE, "descriptor": "Loud Music/Party", "location_type": "Residential Building/House",
               "incident_zip": "11201", "latitude": 40.693412, "longitude": -73.987654}
    with TestClient(create_app()) as app_client:
        assert app_client.post("/predict", json=request).status_code == 200
        assert app_client.post("/predict", json={**request, "latitude": 140.7}).status_code == 422
    for handler in logging.getLogger("civicai.predictions").handlers:
        handler.flush()
    lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1                                                       # only successful predictions are logged
    record = json.loads(lines[0])
    assert record["predicted_class"] in (0, 1) and record["borough"] == "BROOKLYN" and record["has_coordinates"] is True
    for secret in ["40.693412", "73.987654", "11201", "Loud Music", "Residential Building"]:
        assert secret not in lines[0]


def test_request_model_matches_model_meta_input_fields():
    wanted = {f["name"]: f["required"] for f in META["input_fields"]}
    have = {name: field.is_required() for name, field in PredictRequest.model_fields.items()}
    assert have == wanted


def test_requirements_api_pins_the_versions_of_model_meta():
    lines = [l.strip() for l in (ROOT / "requirements-api.txt").read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.startswith("#")]
    pins = dict(l.split("==") for l in lines)
    for name in ["scikit-learn", "xgboost", "pandas", "numpy", "joblib"]:
        assert pins[name] == META["versions"][name], name
    assert {"fastapi", "uvicorn", "pydantic"} <= set(pins)
    assert not {"pytest", "httpx", "jupyter", "matplotlib", "seaborn", "requests"} & set(pins)
