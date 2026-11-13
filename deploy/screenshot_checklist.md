# Screenshot checklist (evidence for sections 8, 9 and 10 of the project document)

Take the screenshots **in this order** while you follow `deploy/ec2_setup.md` (the "Runbook step" column
tells you when). Save them in `docs/screenshots/` with the suggested file names (PNG; on Windows press
**Windows + Shift + S**, select the area, then paste into Paint or save from the notification).

## Rules for every screenshot

* **Never show the contents of the `.pem` file** (do not open it in Notepad or run `type`/`cat` on it). Showing its *name* is fine.
* **No secrets.** There are none in this project, so if anything looks like a key, token or password, do not take the picture.
* **The public IP address of the instance is fine to show.** Do not hide it: it proves the server is on the internet.
* In the AWS console **hide or crop** the 12-digit **account ID** and your e-mail address (top-right account menu, billing pages).
* Your own address in the SSH rule ("My IP") is personal; blurring it is optional.
* The whole window title / address bar should be visible where it proves something (for example the address bar for the Swagger page).
* If a screenshot accidentally shows something it should not, delete it and take it again.

## Section 8: Create EC2 instance

| # | What to capture | Suggested file name | Runbook step | Document placeholder it fills | It must show |
| --- | --- | --- | --- | --- | --- |
| 1 | The AMI selection: Ubuntu Server 22.04 LTS chosen | `section08_01_ami.png` | A3 | "AMI" | The name "Ubuntu Server 22.04 LTS", 64-bit (x86), the "Free tier eligible" label if present |
| 2 | The instance type | `section08_02_instance_type.png` | A4 | "instance type" | `t3.micro`, 2 vCPU, 1 GiB memory (and the free-tier label if present) |
| 3 | The key pair | `section08_03_key_pair.png` | A5 | "key pair (name only, no key contents)" | The "Create key pair" dialog (name `civicai-key`, RSA, .pem) **or** the *Key Pairs* list page. Not the file contents |
| 4 | The security group rules | `section08_04_security_group.png` | A6 | "security group rules" | Exactly two inbound rules: SSH 22 from My IP, Custom TCP 8000 from 0.0.0.0/0 |
| 5 | The running instance | `section08_05_instance_running.png` | A8, A9 | "running instance" | Instance state **Running**, 2/2 checks passed, instance type, **Public IPv4 address** |

## Section 9: Deploy model on EC2

| # | What to capture | Suggested file name | Runbook step | Document placeholder it fills | It must show |
| --- | --- | --- | --- | --- | --- |
| 6 | Logging in with ssh | `section09_01_ssh_login.png` | B1 or B4 | extra figure (no placeholder yet): add it at the start of section 9 | PowerShell with the `ssh -i ... ubuntu@<ip>` line and the Ubuntu welcome text / `ubuntu@ip-...` prompt |
| 7 | The project files on the server | `section09_02_files_on_server.png` | B3 | "files copied to the server" | The end of the copy script's output (file list, "SHA-256 equal"), or `ls -lR ~/civicai/app ~/civicai/src ~/civicai/models` on the server. No `data/`, `notebooks/` or `.pem` in the listing |
| 8 | The packages installed | `section09_03_pip_install.png` | B5, B6 | "requirements installed in the virtual environment" | `python3 --version` (3.10.x), the last lines of `pip install`, and the two version lines of B6 that match |
| 9 | The foreground test (optional but the document has a slot for it) | `section09_04_uvicorn_foreground.png` | B7 | "uvicorn running" | `Model 3.0 loaded` and `Uvicorn running on http://0.0.0.0:8000` |
| 10 | The service is active | `section09_05_systemd_active.png` | B9 | "systemd service active" | `systemctl status civicai --no-pager`: **active (running)** and **enabled** |
| 11 | The service log | `section09_06_journal.png` | B9 | extra figure | `journalctl -u civicai -n 30 --no-pager` showing the model loaded and a few requests |
| 12 | The reboot test | `section09_07_reboot_test.png` | B11 | extra figure | After the reboot: `uptime -p` (a short time) and `systemctl status civicai` **active (running)** with no manual start |
| 13 | `/health` from your laptop | `section09_08_health_from_laptop.png` | B7 or B10 | "GET /health from the laptop" | PowerShell: `Invoke-RestMethod "http://<public-ip>:8000/health"` with `status: ok`, `model_loaded: True`, `model_version: 3.0` |

## Section 10: Test model using Postman (API call)

Set the Postman collection variable `base_url` to `http://<public-ip>:8000` first (collection > Variables).

| # | What to capture | Suggested file name | Runbook step | Document placeholder it fills | It must show |
| --- | --- | --- | --- | --- | --- |
| 14 | Postman: health | `section10_01_postman_health.png` | after B10 | extra figure | Request `1 Health`, status 200, the response body and the test results (all passed) |
| 15 | Postman: a valid request | `section10_02_postman_valid_request.png` | after B10 | "Postman: valid request" | Request `3 Predict - valid: Blocked Driveway` with its JSON body and the public URL |
| 16 | Postman: the response | `section10_03_postman_valid_response.png` | after B10 | "Postman: response" | Status 200, `predicted_class`, `probability_resolved_within_7d`, `decision_threshold 0.5`, `model_version 3.0`, `warnings []` |
| 17 | Postman: an invalid request and its error | `section10_04_postman_invalid_422.png` | after B10 | "Postman: invalid request and its error" | Request `5 Predict - invalid: missing borough` (or `7 ... impossible latitude`), status **422**, and `errors` naming the field |
| 18 | The Swagger page on the public address | `section10_05_swagger_docs.png` | after B10 | extra figure | Browser address bar `http://<public-ip>:8000/docs` and the list of routes (`/health`, `/model-info`, `/predict`) |
| 19 | The smoke test output | `section10_06_smoke_test.png` | B10 | extra figure | PowerShell: `python deploy/smoke_test.py http://<public-ip>:8000` and the PASS table ending in `RESULT: PASS` |

## Before you hand in

* 19 screenshots in `docs/screenshots/`, named as above (5 for section 8, 8 for section 9, 6 for section 10).
* None shows a `.pem` file's contents, an AWS account ID or your e-mail address.
* The instance is **stopped** (or terminated) after the demonstration, and the port 8000 rule is deleted after grading.
