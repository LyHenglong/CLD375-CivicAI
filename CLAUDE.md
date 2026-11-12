# CivicAI - Claude Code Execution Guide

Nov 10, 2026 · @Penguin

## 1. Mission

You are building CivicAI, a university project due **Thursday, November 26, 2026, 11:59 pm**: a classifier that predicts whether an NYC 311 service request will be resolved within a set time, deployed on AWS EC2 behind a FastAPI backend.

The student must be able to explain every step in writing and in a demo, so write simple, readable code with short comments, and explain what you did after each phase.

Copy this document into a file named `CLAUDE.md` in the project root so it is available at the start of each session. The full proposal this guide implements is the document "CivicAI: NYC 311 Resolution Prediction - Project Proposal".

**The assignment requires five demonstrable components:**

1. Data cleaning and preprocessing, with charts and stated decisions.
2. Feature selection, with the target and a justification for each feature.
3. Model selection, training and testing, with a metric comparison.
4. A meta-classifier (stacking) compared against its base models.
5. Deployment on EC2 with an API, tested end to end.

## 2. Working rules

These rules override convenience. If a rule conflicts with a task, stop and ask the student.

**Process**

- Work one phase at a time (Section 5). At the end of each phase, run the checks, summarize what you did and what the numbers were, then **stop and wait for the student to say continue**.
- Run your code and show real output. Never state a result you did not produce. If something fails, show the error and the fix.
- Make small, reviewable changes. Commit to Git at the end of each phase with a clear message.
- Fix random seeds (`random_state=42`) so results are repeatable.
- Put explanations the student can reuse in the report into notebook markdown cells.
- **Project document.** `docs/CivicAI_Project_Document.docx` is built from the course template (`docs/screenshots/DE-ML Final Project Document Template-1.docx`, never edited) by `python docs/build_document.py` (needs `docs/requirements-docs.txt`). At the end of each later phase, update the matching sections in `docs/build_document.py` from saved, **committed** files only (`git show HEAD:<file>`; every typed number is checked against its source file), remove the "TO BE COMPLETED" marker only for sections that are finished, rebuild (the script refreshes the TOC page numbers when Word is installed), and commit the document with the phase. Sections 1 to 6 and 12 are done. Section 7 is filled in Phase 6, sections 8 and 9 in Phase 8, section 10 in Phase 9, section 11 in Phase 10. The student fills in the member name on the cover and the screenshots and Colab link; tell them to re-check the TOC page numbers in Word at the end.

**Data and modeling**

- Never use `closed_date`, `resolution_description`, `status` or any later update column as a model input. They are only used to compute the target.
- Fit every transformer and every target-derived statistic on the training split only.
- Split by time (oldest requests train, newest test). Never shuffle across time.
- Tune and compare models with the training and validation data. Touch the test set once, at the end.
- If the data pull fails or returns less than expected, say so. Never invent or simulate data.
- If a decision would change the target variable or threshold, ask the student first.

**Security and cost**

- Never read, print, write or commit `.pem` files, AWS keys, tokens or `.env` files. Keep them in `.gitignore`.
- Do not run `aws`, `ssh` or `scp` commands, and do not create or start AWS resources. Write the commands into `deploy/ec2_setup.md` for the student to run.
- Do not install system-wide packages. Use the project virtual environment only.

**Code quality**

- Pin versions in `requirements.txt` after the code works. The same scikit-learn version must be used locally and on EC2.
- Validate all API input and return clear error messages.
- Keep dependencies small so the API installs quickly on a small server.

## 3. Project facts

These decisions are fixed unless the student changes them.

| Item | Decision |
| --- | --- |
| Task | Binary classification: will a request be resolved within the threshold? |
| Target | `resolved_within_7d` = 1 if `closed_date` minus `created_date` is 7 days or less, else 0 |
| Threshold rule | 7 days, confirmed by the student after Phase 2. The smaller class (not resolved within 7 days) is 17.2% of training rows, so models use class weights and are judged on macro F1 and ROC-AUC. Do not change the threshold without asking the student |
| Data source | NYC Open Data "311 Service Requests from 2020 to Present", dataset id `erm2-nwe9`, via the Socrata API |
| API pattern | `https://data.cityofnewyork.us/resource/erm2-nwe9.json` with `$limit`, `$offset`, `$where` and `$order`; confirm parameter names in the Socrata documentation before relying on them |
| Sample | 150,000 to 300,000 closed requests, spread across months, saved unchanged to `data/raw/` |
| Final features (12) | `complaint_type`, `descriptor`, `location_type`, `open_data_channel_type`, `borough`, `incident_zip`, `latitude`, `longitude`, and `hour`, `weekday`, `month`, `is_weekend` derived from `created_date`. Chosen in Phase 4 (`notebooks/02_feature_selection.ipynb`); the list is `FINAL_FEATURES` in `src/features.py`. `agency`, `address_type` and `community_board` were dropped because an API caller cannot supply them |
| Excluded columns | `closed_date` (target source only), `resolution_description`, `status`, later update timestamps |
| Split | By `created_date`, 198,867 cleaned rows: train 139,206 rows (2024-01-01 00:00 to 2025-12-02 20:45), validation 29,830 rows (2025-12-02 20:48 to 2026-04-14 21:10), test 29,831 rows (2026-04-14 21:11 to 2026-08-31 23:59). Saved as `train.parquet`, `val.parquet` and `test.parquet` in `data/processed/` |
| Models | Two baselines (majority class; complaint type only) and six candidates: Logistic Regression, Decision Tree, Random Forest, Extra Trees, gradient boosting with `XGBClassifier` (the pipeline output is sparse, which `HistGradientBoostingClassifier` cannot take) and Bernoulli Naive Bayes. Tuned settings are in `reports/best_settings.json`; results are in `reports/model_comparison.csv` |
| Selected for the stack | **XGBoost, Bernoulli Naive Bayes and Random Forest** (Phase 5b). Rule: rank by mean cross-validated macro F1 on the training split and take the top 3, but skip a model whose validation probability correlation with an already chosen model is above 0.97. Logistic Regression ranked second but was skipped (0.975 with XGBoost). Best single model: XGBoost (validation macro F1 0.835, ROC-AUC 0.943) |
| Meta-classifier | `StackingClassifier`, final estimator Logistic Regression, 5-fold CV, probability outputs. Phase 6 (v1) used the three selected models (XGBoost, Bernoulli Naive Bayes, Random Forest); Phase 6b (v2, deployed) added Logistic Regression back and a tuned cut-off. Bernoulli Naive Bayes needs the all-0/1 encoder (`build_binary_encoder` in `src/features.py`), so each base model carries its own encoder after the shared cleaning step |
| Metrics | Accuracy, precision, recall, macro F1, ROC-AUC, confusion matrix |
| Saved artifact | One scikit-learn `Pipeline` (`clean` step plus the stack) saved with `joblib` as `models/model.joblib`. It takes the RAW API fields and returns a probability. **`models/model.joblib` is gitignored: copy it to EC2 with `scp`.** `models/model_meta.json` (SHA-256, library versions, the 9 raw input fields, split dates, final-estimator settings) and `models/metrics.json` are small text files and are committed |
| Phase 6 model (v1, superseded) | Pipeline(`clean` = `RequestCleaner`, `stack` = `StackingClassifier` over XGBoost, Bernoulli Naive Bayes and Random Forest with their tuned settings from `reports/best_settings.json`; final estimator Logistic Regression `class_weight="balanced"`, `C=0.1`, chosen by validation macro F1; 5-fold stratified shuffled CV inside the training split, because `StackingClassifier` cannot use `TimeSeriesSplit`). Fitted on the training split only, not refitted on train+validation. 19.8 MB (Random Forest 17.95 MB). Stack validation: macro F1 0.828, ROC-AUC 0.943. Stack test (scored once): macro F1 0.806, ROC-AUC 0.936, class 0 recall 0.879, class 0 precision 0.584. **The stack did not beat XGBoost alone** (test macro F1 0.826, ROC-AUC 0.937); it is deployed because the course requires a meta-classifier. Frozen as git tag `phase6-v1` and `models/model_v1.joblib` (gitignored) |
| **Final model (Phase 6b, v2, deployed)** | `models/model.joblib`, version 2.0: Pipeline(`clean` = `RequestCleaner`, `stack` = `FixedThresholdClassifier` (cut-off **0.61** on the probability of class 1, chosen on training out-of-fold predictions) around a `StackingClassifier` over **XGBoost, Logistic Regression, Bernoulli Naive Bayes and Random Forest**; final estimator Logistic Regression without class weights, `C=0.1`). `model.predict()` applies the cut-off; `predict_proba()` is unchanged; use `src.features.get_stack(model)` to reach the stack. Rebuilt by `tr.build_stack_v2` / `python -m src.train`. Fitted on the training split only. 19.8 MB, about 90-140 ms per prediction on the laptop. Validation: macro F1 0.834, ROC-AUC 0.943. **Clean forward test: macro F1 0.807, ROC-AUC 0.933, class 0 recall 0.779, class 0 precision 0.608.** The success criterion (at least XGBoost alone on validation macro F1) was **not met** (-0.12 points); on the forward test the stack and XGBoost alone are level (0.8074 vs 0.8070). It is deployed because the course requires a meta-classifier |
| Test sets and which numbers are clean | (1) **Forward test** `data/processed/test_forward.parquet`: 12,069 closed requests created 2026-09-01 to 2026-10-28, pulled with `python src/fetch_data.py --start 2026-09-01 --end 2026-10-29 --name forward`, scored once in Phase 6b: **clean, use these in the report** (`reports/test_forward_results.csv`, rows "forward test (clean)"). Closed requests only, so somewhat easier than reality (9.2% of that period was still open); the rows "forward test + still-open requests" add those as class 0. (2) **Original test split**: the v1 results in `reports/test_results.csv` are clean for v1; the v2 result on it is labelled "reused, optimistic" and is not clean. (3) Validation scores of v2 are slightly optimistic (best of 7 experiments). **Both test sets are now used up: never use either to choose anything** |
| Model input fields | 9 raw fields, not 12: required `complaint_type`, `borough`, `open_data_channel_type`, `created_at`; optional `descriptor`, `location_type`, `incident_zip`, `latitude`, `longitude`. The model's 12 features include hour, weekday, month and `is_weekend`, which the pipeline derives from `created_at`. Full list with rules in `models/model_meta.json` |
| Backend | FastAPI with Uvicorn on Ubuntu EC2 |
| Python | 3.10 or newer; record exact versions in `models/model_meta.json` |

## 4. Repository structure

Create this layout in Phase 0 and keep to it. Notebooks hold the exploration the student presents; `src/` holds the reproducible code; `app/` is the only part that runs on EC2.

```text
civicai-ec2/
  CLAUDE.md
  README.md
  requirements.txt
  .gitignore              # data/, models/, .env, *.pem, __pycache__/
  data/
    raw/                  # unchanged API pull
    processed/            # cleaned data and splits
    sample_requests.json  # 5 real validation-split requests with the model's answers, and 3 invalid ones with the expected 422
  notebooks/
    01_data_cleaning_eda.ipynb
    02_feature_selection.ipynb
    03_model_comparison.ipynb
    04_stacking.ipynb
    05_stacking_improvement.ipynb   # Phase 6b
  src/
    fetch_data.py
    preprocessing.py
    features.py
    train.py
    evaluate.py           # not created: evaluation functions live in train.py
    check_saved_model.py  # fresh-process check of models/model.joblib (not needed by the API)
  models/
    model.joblib          # v2, gitignored; copy to EC2 with scp
    model_v1.joblib       # Phase 6 model, gitignored; not needed on the server
    metrics.json
    model_meta.json
    cleaning_values.json  # not needed on the server
  reports/
    figures/
    summary.md
  app/
    main.py
    schemas.py
    predictor.py
    streamlit_app.py      # optional
  tests/
    test_api.py
  deploy/
    civicai.service
    ec2_setup.md
  docs/
    screenshots/
```

## 5. Build phases

Do the phases in order. Each phase ends with its acceptance check, a short summary, a Git commit, and a stop for the student's approval.

| Phase | Target dates | What to do | Done when |
| --- | --- | --- | --- |
| 0. Setup | Nov 10 | Create the layout in Section 4, a virtual environment, `.gitignore`, and `requirements.txt` with pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib, pyarrow, jupyter, fastapi, uvicorn, pydantic, requests, pytest and httpx; run `git init` | `pip install -r requirements.txt` succeeds and the folders exist |
| 1. Data pull | Nov 10-11 | Write `src/fetch_data.py`: page through the Socrata API with `status = Closed` and a date range, retry on errors, resume if interrupted, save unchanged to `data/raw/` | File holds at least 150,000 rows; printed row count, date range and nulls per column |
| 2. Target and threshold check | Nov 12 | Compute duration in hours, drop rows with missing or negative durations, plot the duration histogram and class balance for the 7-day target | Class shares printed and a threshold recommendation made; **stop and ask the student to confirm** |
| 3. Cleaning and EDA | Nov 12-14 | In notebook 01, check and handle missing values, duplicates by `unique_key`, outliers, inconsistent text, wrong types; save every chart to `reports/figures/`; keep a cleaning log table | Each issue has a check, a chart and a stated fix; `data/processed/clean.parquet` saved |
| 4. Features and pipeline | Nov 15-16 | Write `src/features.py` and `src/preprocessing.py`; derive hour, weekday, month, weekend flag; build a `ColumnTransformer` in a `Pipeline`; make the chronological split; rank relevance in notebook 02 with mutual information and tree importance | Pipeline fits on train only; split date ranges printed; an assert confirms no excluded column is in the inputs; kept and dropped features listed with reasons |
| 5. Model comparison | Nov 17-18 | In notebook 03 and `src/train.py`, train the two baselines and six candidate models (Logistic Regression, Decision Tree, Random Forest, Extra Trees, XGBoost, Bernoulli Naive Bayes), cross-validate on train with `TimeSeriesSplit(5)`, tune lightly, score on validation, save `reports/model_comparison.csv`, ROC curves and confusion matrices, and select 3 models for the stack with the rule in Section 3 | One table compares all models on the same split; best single model named with the metric used; the 3 selected models marked in `selected_for_stack` (done: XGBoost, Bernoulli Naive Bayes, Random Forest) |
| 6. Stacking | Nov 18-19 | In notebook 04 and `src/train.py` (`build_stack`; `python -m src.train` rebuilds it and checks it against the saved results without reading the test split), build the `StackingClassifier`, compare it with each base model, score the test set once, save `models/model.joblib`, `metrics.json` and `model_meta.json`, and `data/sample_requests.json` | Stack versus base models table done (`reports/stacking_validation.csv`, `reports/test_results.csv`); the saved pipeline loads in a fresh Python process and predicts the sample requests (`src/check_saved_model.py`). Done: the stack did not beat XGBoost alone, and is deployed anyway |
| 6b. Stack improvement | Nov 12 | In notebook 05: freeze v1 (tag `phase6-v1`); build a fresh forward test set; diagnose the gap to XGBoost (it was the decision threshold: 0.73 points at 0.5, 0.08 with tuned cut-offs); run 7 experiments logged in `reports/stacking_experiments.csv` (v1 members with a tuned cut-off, with and without class weights; XGBoost+LR+NB; XGBoost+LR+NB+RF; XGBoost+LR; two XGBoost depths; a gradient boosting final estimator); pick the winner by validation macro F1; refit on train only; score the forward test once | Done: winner XGBoost+LR+NB+RF, cut-off 0.61. Success criterion not met on validation (-0.12 points vs XGBoost alone); level with XGBoost alone on the clean forward test; +3.0 points over v1 |
| 7. API | Nov 20-21 | Build the FastAPI app as specified in Section 6 and the tests in `tests/test_api.py` | `pytest` passes; `uvicorn app.main:app` runs locally; `/docs` shows the routes |
| 8. Deployment files | Nov 21-22 | Write `deploy/civicai.service` and `deploy/ec2_setup.md` (Section 7); the student runs them on EC2 | The student reports that `GET /health` works at the public address |
| 9. End-to-end tests | Nov 23-24 | Make `tests/test_api.py` run against a remote URL from the `API_URL` environment variable; add a Postman collection under `docs/`; list the screenshots the student must capture | Valid and invalid requests behave as in Section 6 against the live EC2 API |
| 10. Report support | Nov 24-25 | Write `reports/summary.md` with results tables, figure list and limitations, and a `README.md` with run steps | Every number in the summary comes from a saved results file |

## 6. API specification

The API accepts a request's raw fields, derives hour, weekday, month and weekend flag on the server with the same code used in training, and returns the model's prediction.

| Route | Purpose |
| --- | --- |
| `GET /health` | Returns 200 with the model version when the model is loaded |
| `POST /predict` | Takes one request as JSON and returns the prediction |
| `GET /docs` | Swagger UI generated by FastAPI |

**Example request**

```json
{
  "complaint_type": "Noise - Residential",
  "descriptor": "Loud Music/Party",
  "borough": "BROOKLYN",
  "location_type": "Residential Building/House",
  "open_data_channel_type": "PHONE",
  "incident_zip": "11201",
  "latitude": 40.69,
  "longitude": -73.99,
  "created_at": "2026-11-09T22:15:00"
}
```

**Example response** (values are illustrative)

```json
{
  "predicted_class": 1,
  "label": "resolved within 7 days",
  "probability_within_threshold": 0.82,
  "threshold_days": 7,
  "model_version": "1.0"
}
```

**Rules**

- Define the input with a Pydantic model in `app/schemas.py`. Required fields: `complaint_type`, `borough`, `open_data_channel_type`, `created_at`. The others are optional and default to "Unknown" or missing.
- Check ranges: latitude and longitude must be plausible numbers, and `created_at` must be a valid ISO date and time.
- Load `models/model.joblib` once at startup in `app/predictor.py`, not on every request. Right after loading, set the Random Forest inside the stack to one thread: `from src.features import get_stack` then `get_stack(model).named_estimators_["random_forest"]["model"].n_jobs = 1`. Answers are identical and one prediction is faster. The pipeline needs the `src` package on the import path, so start the server from the project root.
- **Decision rule.** The predicted class is 1 ("resolved within 7 days") when the probability is at or above the cut-off in `models/model_meta.json` (`label.decision_threshold`, 0.61 for v2). `model.predict()` already applies it, so use `predict()` for `predicted_class` and `predict_proba()[:, 1]` for `probability_within_threshold`; do not compare the probability with 0.5 in the API.
- **The model does not validate input.** The three invalid requests in `data/sample_requests.json` (missing `borough`, text in `latitude`, `latitude` 140.7) still receive a probability if sent to the model directly. The Pydantic model must reject them with 422 and name the field. Use the valid requests in the same file as test cases: their probabilities must equal `expected_probability_within_threshold`.
- Missing or invalid fields return 422 with the field named. A category the model never saw still returns a prediction, because the encoder ignores unknown values.
- If the model fails to load, `/health` and `/predict` return 503 with a clear message. Never return a Python stack trace to the client.
- Log each request and prediction to a local file for later review.

## 7. EC2 deployment

You write the files and the step-by-step commands. The student runs them on their own AWS account, because you must not hold AWS credentials or key files.

**Write `deploy/ec2_setup.md` with these steps, each with the exact command:**

1. Launch an Ubuntu EC2 instance. Start small, and check the model file size and memory use before choosing. Check current pricing and free-tier terms first.
2. Create a key pair and a security group: SSH only from the student's IP, and one TCP port for the API (8000).
3. Connect over SSH and install `python3-venv` and `python3-pip`.
4. Copy `app/`, `src/__init__.py`, `src/preprocessing.py`, `src/features.py`, `models/model.joblib`, `models/model_meta.json` and `requirements.txt` to the server with `scp`, keeping the same folder layout. The saved pipeline needs the three `src/` files to load. `models/cleaning_values.json` is not needed on the server. `models/model.joblib` is not in Git, so it must be copied with `scp` (about 20 MB). To check the copy on the server, also copy `src/check_saved_model.py` and `data/sample_requests.json` and run `python -I src/check_saved_model.py` from the project root; it verifies the SHA-256 and the library versions in `model_meta.json` and the five sample probabilities.
5. Create a virtual environment on the server and install the pinned requirements. The scikit-learn version (1.7.2) must match `model_meta.json`; use the same xgboost, pandas, numpy and joblib versions too.
6. Test by running `uvicorn app.main:app --host 0.0.0.0 --port 8000` from the project root (so that `src` can be imported), then call `GET /health` from the student's laptop using the instance's public address.
7. Install the `systemd` service below so the API restarts and survives closing the terminal.
8. Stop the instance when not testing. Its public address changes after a stop and start unless an Elastic IP is attached, so update `API_URL` each time.

**Write `deploy/civicai.service`:**

```ini
[Unit]
Description=CivicAI API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/civicai
ExecStart=/home/ubuntu/civicai/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

After the student confirms the instance is running, you may call the public `/health` and `/predict` routes with `curl` or `pytest` using the `API_URL` the student provides. Do not touch AWS itself.

## 8. Ready-to-paste prompts

Start each session in the project folder. Paste one prompt at a time and review the result before sending the next.

**Start of the project**

```text
Read CLAUDE.md fully. Do Phase 0 only: create the folder layout, virtual environment, .gitignore, requirements.txt and git repo. Run pip install and show that it works. Then summarize and stop.
```

**Phases 1 and 2: data and threshold**

```text
Do Phase 1: write src/fetch_data.py to pull closed NYC 311 requests from the Socrata API into data/raw/. Print row count, date range and nulls per column. Then do Phase 2: compute the target, show class balance and a duration histogram, and recommend the threshold. Stop and ask me to confirm.
```

**Phase 3: cleaning and EDA**

```text
Do Phase 3 in notebooks/01_data_cleaning_eda.ipynb. For missing values, duplicates, outliers, inconsistent values and wrong types: show the check, a chart saved to reports/figures/, and the fix, with a short markdown explanation of each decision. Save data/processed/clean.parquet and a cleaning log table. Stop when done.
```

**Phase 4: features and pipeline**

```text
Do Phase 4. Build src/features.py and src/preprocessing.py with a ColumnTransformer inside a Pipeline, make the chronological split, and rank feature relevance in notebooks/02_feature_selection.ipynb. Add an assert that no excluded column is used as an input. Show the kept and dropped features with reasons. Stop when done.
```

**Phases 5 and 6: models and stacking**

```text
Do Phase 5: train the majority baseline, Logistic Regression, Random Forest and gradient boosting, cross-validate on the training split, and compare them on the validation split in one table. Stop and show me. After I approve, do Phase 6: build the StackingClassifier, compare it with each base model, score the test set once, save models/model.joblib, metrics.json and model_meta.json, and prove the saved pipeline loads in a fresh process.
```

**Phase 7: API**

```text
Do Phase 7 using Section 6 of CLAUDE.md: app/main.py, schemas.py, predictor.py, and tests/test_api.py. Run pytest and start the server locally. Show a valid request, a missing-field request and an invalid-value request with their responses.
```

**Phases 8 and 9: deployment files and live tests**

```text
Do Phase 8: write deploy/ec2_setup.md and deploy/civicai.service. Do not run any AWS, ssh or scp command. Stop. After I confirm the API is running on EC2, I will give you API_URL; then do Phase 9: run tests against it, create the Postman collection, and list the screenshots I must take.
```

**Phase 10: report support**

```text
Do Phase 10: write reports/summary.md with results tables taken only from saved results files, the list of figures and the limitations, and write README.md with run steps. Do not write the final report prose.
```

## 9. Definition of done

The project is finished when every box is ticked and the student can explain each item. Tick them only after checking the evidence.

**Data cleaning and preprocessing**

- [ ] Raw sample saved unchanged; row count and date range recorded
- [ ] Missing values, duplicates, outliers, inconsistent values and wrong types each checked, charted and handled
- [ ] Cleaning log table and saved charts in `reports/figures/`

**Feature selection**

- [ ] Target defined and class balance shown
- [ ] Each kept and dropped feature listed with a reason and a relevance measure
- [ ] Assert in code that no excluded column reaches the model

**Model selection and meta-classifier**

- [ ] Baseline plus three models compared in one table on the same split
- [ ] `StackingClassifier` compared with each base model, with F1, ROC-AUC and confusion matrices
- [ ] Test set scored once; results saved in `models/metrics.json`

**EC2 deployment and testing**

- [ ] Saved pipeline loads in a fresh process and predicts
- [ ] API running on EC2 under `systemd`; `/health` answers from the student's laptop
- [ ] Tests pass locally and against `API_URL`: valid, edge, missing field, wrong value
- [ ] Postman collection and screenshots saved in `docs/`

**Hand-off**

- [ ] `README.md` with run steps and `reports/summary.md` with results
- [ ] No secrets, keys or raw data in Git
- [ ] EC2 instance stopped after the demonstration
