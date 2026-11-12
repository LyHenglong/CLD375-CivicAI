"""Fresh-process check of the saved model (Phase 6). Not needed by the API.

Run it in a clean Python process from anywhere, giving the project root:
    python -I src/check_saved_model.py            # project root = the folder above src/
    python -I src/check_saved_model.py /path/to/project

It loads models/model.joblib, runs every request in data/sample_requests.json
as a raw dict, and checks that
  - the file matches the SHA-256 and the library versions in models/model_meta.json,
  - the probabilities equal the ones stored in sample_requests.json (made by the notebook),
  - nothing from the notebooks was needed.
The same script can be run on the EC2 server after copying the files there.
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))          # the saved pipeline needs the "src" package (python -I does not add it)

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

problems = []


def check(ok, message):
    print(("PASS  " if ok else "FAIL  ") + message)
    if not ok:
        problems.append(message)


meta = json.loads((root / "models" / "model_meta.json").read_text(encoding="utf-8"))
samples = json.loads((root / "data" / "sample_requests.json").read_text(encoding="utf-8"))
model_path = root / "models" / "model.joblib"

print(f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, "
      f"pandas {pd.__version__}, numpy {np.__version__}, joblib {joblib.__version__}")
print(f"Project root: {root}\n")

# 1. The file is the one that was evaluated
digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
check(digest == meta["sha256"], f"SHA-256 of model.joblib matches model_meta.json ({digest[:16]}...)")
size_mb = model_path.stat().st_size / 1e6
check(size_mb < 60, f"file size {size_mb:.1f} MB is below 60 MB")

# 2. Library versions (scikit-learn must match; the others only warn)
versions = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__,
            "numpy": np.__version__, "joblib": joblib.__version__, "python": platform.python_version()}
for name, version in versions.items():
    same = version == meta["versions"][name]
    if name == "scikit-learn":
        check(same, f"scikit-learn {version} equals the version in model_meta.json ({meta['versions'][name]})")
    elif not same:
        print(f"WARN  {name} is {version}, model_meta.json says {meta['versions'][name]}")

# 3. Load and predict the sample requests as raw dicts, exactly as the API will
model = joblib.load(model_path)
loaded = sorted(m for m in sys.modules if m == "src" or m.startswith("src."))
check(all(not m.startswith(("notebook", "ipykernel", "__main__")) for m in loaded) and loaded == ["src", "src.features", "src.preprocessing"],
      f"only the project modules were needed: {loaded}")

for item in samples["valid_requests"]:
    probability = float(model.predict_proba(pd.DataFrame([item["request"]]))[0, 1])
    predicted = int(model.predict(pd.DataFrame([item["request"]]))[0])
    same_probability = abs(probability - item["expected_probability_within_threshold"]) < 1e-9
    check(same_probability and predicted == item["expected_predicted_class"],
          f"{item['name']}: probability {probability:.6f} (notebook {item['expected_probability_within_threshold']:.6f}), "
          f"class {predicted}, true label {item['true_label']}")

batch = pd.DataFrame([item["request"] for item in samples["valid_requests"]])
batch_probability = model.predict_proba(batch)[:, 1]
expected = np.array([item["expected_probability_within_threshold"] for item in samples["valid_requests"]])
check(np.abs(batch_probability - expected).max() < 1e-9, "the five requests sent together give the same probabilities")

# 4. For information: the model itself does not validate input. The API (Phase 7) must reject these.
print("\nInvalid sample requests sent to the model directly (the API must reject them with the status shown):")
for item in samples["invalid_requests"]:
    try:
        probability = float(model.predict_proba(pd.DataFrame([item["request"]]))[0, 1])
        outcome = f"the model still answers {probability:.3f}"
    except Exception as error:                        # noqa: BLE001 - only reporting
        outcome = f"the model raises {type(error).__name__}"
    print(f"  {item['name']}: expected API status {item['expected_status']} on '{item['expected_error_field']}'; {outcome}")

print()
if problems:
    print(f"{len(problems)} check(s) FAILED")
    sys.exit(1)
print("All checks passed.")
