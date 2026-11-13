"""Smoke test of a running CivicAI API. Standard library plus `requests` only.

Run it from your own computer against the local server or against EC2:

    python deploy/smoke_test.py http://127.0.0.1:8000
    python deploy/smoke_test.py http://<public-ip-of-the-server>:8000

It checks, in order:
  1. GET /health          the model is loaded and the library versions match
  2. GET /model-info      the model description is served
  3. POST /predict        every valid request of data/sample_requests.json: the probability
                          must equal the saved value (to 6 decimals) and so must the class
  4. POST /predict        every invalid request of that file: 422 and the field is named
  5. GET /docs            the Swagger page is there

It prints a PASS/FAIL table and exits with 0 only if every check passed (1 otherwise),
so it can also be used in a script. It sends only the sample requests (real requests of
the validation split); it never sends anything else and needs no keys or passwords.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:                                           # pragma: no cover
    sys.exit("This script needs the 'requests' package: pip install requests")

DEFAULT_SAMPLES = Path(__file__).resolve().parents[1] / "data" / "sample_requests.json"
TOLERANCE = 1e-6                                              # the API rounds probabilities to 6 decimals

rows = []                                                     # one row per check: (check, expected, got, ms, passed)


def short(text, width):
    text = str(text)
    return text if len(text) <= width else text[: width - 3] + "..."


def record(check, expected, got, milliseconds, passed):
    rows.append((check, expected, got, milliseconds, bool(passed)))


def timed(call):
    """Run call() and return (result, milliseconds)."""
    start = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - start) * 1000


def print_table(title):
    widths = (44, 26, 30)
    print()
    print(title)
    print("-" * 118)
    print(f"{'#':>2}  {'Check':<{widths[0]}} {'Expected':<{widths[1]}} {'Got':<{widths[2]}} {'ms':>6}  Result")
    print("-" * 118)
    for number, (check, expected, got, milliseconds, passed) in enumerate(rows, 1):
        print(f"{number:>2}  {short(check, widths[0]):<{widths[0]}} {short(expected, widths[1]):<{widths[1]}} "
              f"{short(got, widths[2]):<{widths[2]}} {milliseconds:>6.0f}  {'PASS' if passed else 'FAIL'}")
    print("-" * 118)


def field_names(response):
    try:
        return [problem["field"] for problem in response.json().get("errors", [])]
    except (ValueError, AttributeError, KeyError, TypeError):
        return []


def main():
    parser = argparse.ArgumentParser(description="Smoke test of a running CivicAI API.")
    parser.add_argument("base_url", nargs="?", default="http://127.0.0.1:8000", help="for example http://127.0.0.1:8000 or http://<public-ip>:8000")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="path of sample_requests.json (default: data/sample_requests.json of this project)")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for each answer (default 30)")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    try:
        samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        sys.exit(f"Cannot read the sample requests ({args.samples}): {error}")

    session = requests.Session()

    def get(path):
        return session.get(base + path, timeout=args.timeout)

    def post(body):
        return session.post(base + "/predict", json=body, timeout=args.timeout)

    # ------------------------------------------------------------------ 1. health
    try:
        response, ms = timed(lambda: get("/health"))
    except requests.exceptions.RequestException as error:
        record("GET /health", "200, model loaded", f"no answer ({type(error).__name__})", 0, False)
        print_table(f"CivicAI smoke test against {base}")
        print("\nThe server did not answer. Check, in this order: is the instance running? is the URL right"
              " (http, not https; the port 8000)? does the security group allow port 8000? is the service running"
              " (systemctl status civicai)?  See the troubleshooting table in deploy/ec2_setup.md.")
        return 1
    body = {}
    try:
        body = response.json()
    except ValueError:
        pass
    ok = response.status_code == 200 and body.get("status") == "ok" and body.get("model_loaded") is True
    record("GET /health: model loaded", "200, status ok, loaded", f"{response.status_code}, {body.get('status')}, version {body.get('model_version')}", ms, ok)
    record("GET /health: library versions match", "versions_match = True", f"{body.get('versions_match_model_meta')} {body.get('version_mismatches') or ''}".strip(), 0,
           body.get("versions_match_model_meta") is True)

    # ------------------------------------------------------------------ 2. model info
    response, ms = timed(lambda: get("/model-info"))
    try:
        info = response.json()
    except ValueError:
        info = {}
    ok = (response.status_code == 200 and "model_version" in info and isinstance(info.get("input_fields"), list)
          and "decision_threshold" in info.get("label", {}))
    record("GET /model-info", "200, version + fields + cut-off", f"{response.status_code}, version {info.get('model_version')}, cut-off {info.get('label', {}).get('decision_threshold')}", ms, ok)

    # ------------------------------------------------------------------ 3. valid requests
    for number, item in enumerate(samples["valid_requests"], 1):
        request = item["request"]
        expected_probability = item["expected_probability_within_threshold"]
        expected_class = item["expected_predicted_class"]
        name = f"POST /predict valid {number}: {request['complaint_type'][:22]}"
        try:
            response, ms = timed(lambda: post(request))
            answer = response.json()
        except (requests.exceptions.RequestException, ValueError) as error:
            record(name, f"p={expected_probability:.6f}", f"no usable answer ({type(error).__name__})", 0, False)
            continue
        probability = answer.get("probability_resolved_within_7d")
        same = (response.status_code == 200 and isinstance(probability, (int, float)) and abs(probability - expected_probability) <= TOLERANCE
                and answer.get("predicted_class") == expected_class and answer.get("warnings") == [])
        record(name, f"p={expected_probability:.6f}, class {expected_class}",
               f"{response.status_code}, p={probability}, class {answer.get('predicted_class')}", ms, same)

    # ------------------------------------------------------------------ 4. invalid requests
    for item in samples["invalid_requests"]:
        wanted = item["expected_error_field"]
        name = f"POST /predict invalid: {item['name']}"
        try:
            response, ms = timed(lambda: post(item["request"]))
        except requests.exceptions.RequestException as error:
            record(name, f"422 on {wanted}", f"no answer ({type(error).__name__})", 0, False)
            continue
        fields = field_names(response)
        record(name, f"422 on {wanted}", f"{response.status_code} on {', '.join(fields) or '-'}", ms,
               response.status_code == item["expected_status"] and wanted in fields and "Traceback" not in response.text)

    # ------------------------------------------------------------------ 5. swagger page
    response, ms = timed(lambda: get("/docs"))
    record("GET /docs (Swagger page)", "200, HTML", f"{response.status_code}, {response.headers.get('content-type', '')[:20]}", ms,
           response.status_code == 200 and "html" in response.headers.get("content-type", ""))

    # ------------------------------------------------------------------ result
    print_table(f"CivicAI smoke test against {base}")
    failed = [row for row in rows if not row[4]]
    timings = [row[3] for row in rows if row[3] > 0]
    print(f"{len(rows) - len(failed)} of {len(rows)} checks passed. Time per request: median {statistics.median(timings):.0f} ms, slowest {max(timings):.0f} ms."
          if timings else f"{len(rows) - len(failed)} of {len(rows)} checks passed.")
    print("RESULT: PASS" if not failed else f"RESULT: FAIL ({len(failed)} check(s) failed)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
