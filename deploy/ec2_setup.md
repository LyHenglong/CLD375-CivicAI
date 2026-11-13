# CivicAI on AWS EC2: step-by-step runbook

Follow this file from top to bottom. It is written for someone doing this for the first time.
**Part A** is clicking in the AWS console. **Part B** is typing commands in Windows PowerShell
(on your laptop) and then on the server.

```text
   your laptop                      the internet                    the EC2 server (Ubuntu)
  ┌────────────┐   ssh (port 22)   ┌──────────┐    ┌─────────────────────────────────────────┐
  │ PowerShell │ ───────────────►  │          │ ─► │ systemd ─► uvicorn ─► FastAPI app ─► model │
  │ Postman    │   http (port 8000)│          │    │          (port 8000)   /home/ubuntu/civicai │
  └────────────┘ ───────────────►  └──────────┘    └─────────────────────────────────────────┘
```

---

## 0. Read this first: what costs money

| Thing | Costs money when... | What to do |
| --- | --- | --- |
| The instance (the computer) | it is **running** | **Stop** it whenever you are not testing. **Terminate** it after grading. |
| The disk (8 GiB) | it exists, **even if the instance is stopped** | Terminate the instance at the end (the disk is deleted with it). |
| A public IP address | AWS charges about 0.005 US dollars per hour (about 3.6 dollars a month) for **every public IPv4 address**, including the one that is assigned automatically while the instance runs, and an Elastic IP that is allocated but not attached. Check the current terms on the AWS pricing page. | Do not allocate an Elastic IP unless you need one (section A10). |
| Data leaving AWS | you send a lot of data out | Not an issue for this API (small answers). |

**Do the billing alert (A0) before you launch anything.** The prices in this file are approximate and
change; always check the live numbers on https://aws.amazon.com/ec2/pricing/on-demand/ and on your
own *Billing > Free Tier* page.

**Stop is not Terminate.** Stop = turn the computer off, keep the disk (small cost, can be started again).
Terminate = delete the computer and its disk for ever (cannot be undone).

---

## The choices used in this runbook

| Item | Choice | Why (short) |
| --- | --- | --- |
| Region | Asia Pacific (Singapore) `ap-southeast-1` | Nearest AWS region to Cambodia: fastest for you and for a grader in the region |
| AMI (operating system) | Ubuntu Server **22.04 LTS** (64-bit x86) | Its Python is 3.10, the version the model was built with (`models/model_meta.json`: 3.10.10) |
| Instance type | **t3.micro** (2 vCPU, 1 GiB memory) | The API needs about 160 MB; this is the smallest size that also installs the packages safely |
| Key pair | `civicai-key`, type RSA, format `.pem` | Works with the ssh program built into Windows |
| Security group | `civicai-sg`: SSH 22 from **my IP only**, TCP 8000 from anywhere (demo) | Nothing else is open |
| Disk | default 8 GiB gp3 | Enough: system about 3 GB, Python packages about 1.5 GB |
| Folder on the server | `/home/ubuntu/civicai` | Matches `deploy/civicai.service` |
| Port | 8000 | The API's port |

---

# PART A: in the AWS console (you click, with screenshots)

Screenshots to take are listed in `deploy/screenshot_checklist.md` (it follows the same order).

## A0. Billing alert first (5 minutes)

1. Console search bar: type **Budgets** > *Billing and Cost Management > Budgets* > **Create budget**.
2. Choose **Use a template (simplified)** > **Zero spend budget** (alerts you as soon as anything is charged)
   or **Monthly cost budget** with an amount of **5 USD**. Enter your email. Create.
3. *Billing and Cost Management > Billing preferences*: tick **Receive AWS Free Tier alerts** and
   **Receive CloudWatch billing alerts** (type your email). Save.

*Why:* a forgotten instance is the most common surprise bill. The alert emails you before it grows.

## A1. Region

Top-right corner of the console: click the region name and pick **Asia Pacific (Singapore) ap-southeast-1**.
*Why:* it is the closest region (less delay). *Careful:* every resource lives in one region. If you cannot
find your instance later, look at the region name in the top-right corner first.

## A2. Start the launch wizard

Search bar: **EC2** > **Instances** > orange button **Launch instances**. **Name:** `civicai-api`.

## A3. Operating system (AMI)

**Application and OS Images > Quick Start > Ubuntu.** In the dropdown choose
**Ubuntu Server 22.04 LTS (HVM), SSD Volume Type**, architecture **64-bit (x86)**.
(The wizard may show a newer release first. Open the dropdown or *Browse more AMIs* and pick 22.04.)

*Why 22.04:* the model was built with Python 3.10.10 and the API checks the Python version when it
starts. Ubuntu 22.04 ships Python 3.10.12 (same version family: fine). Ubuntu 24.04 ships Python 3.12,
which would need extra steps (see Part B, step B5, "if the Python version is not 3.10").
*Login name* for Ubuntu images is always `ubuntu`.

## A4. Instance type

Pick **t3.micro** (2 vCPU, 1 GiB memory).

* *Why this size:* the running API uses about 160 MB of memory (measured) and the model file is 1.8 MB.
  Installing the packages needs a few hundred MB for a short time. `t3.nano` (0.5 GiB) is too small;
  `t3.small` (2 GiB, twice the price) is fine but not needed.
* *Free tier:* the wizard puts a **"Free tier eligible"** label next to the types that qualify for
  **your** account. The rules changed in 2025, so look at the label and at *Billing > Free Tier* instead
  of trusting any document:
  * accounts created **before 15 July 2025** usually have **12 months** of 750 free hours per month for
    t2.micro or t3.micro (whichever is labelled in your region);
  * accounts created **on or after 15 July 2025** are on a **Free plan** with credits (about 6 months,
    limited to certain small instance types) instead.
  If t3.micro has no "Free tier eligible" label, use the type that has it (for example `t2.micro`); the
  steps are the same.
* *Cost warning:* outside the free tier, a t3.micro costs roughly 0.011 to 0.013 US dollars per hour
  (about 8 to 10 dollars a month if left on all month), plus about 3.6 dollars a month for the public IP and about 0.7 dollars for the disk. Two hours of
  testing cost a few cents. **Stop the instance when you finish a session.**

## A5. Key pair (your login key)

1. **Key pair (login) > Create new key pair.** Name: `civicai-key`. Type: **RSA**. Format: **.pem**.
2. Click **Create key pair**. Your browser downloads `civicai-key.pem`. **AWS keeps no copy: if you lose
   this file you cannot log in to this instance.**
3. Move it **out of the Downloads folder and out of the project folder** (so it can never be copied or
   committed), into your `.ssh` folder. In PowerShell:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh" | Out-Null
Move-Item "$env:USERPROFILE\Downloads\civicai-key.pem" "$env:USERPROFILE\.ssh\civicai-key.pem"
```

4. **Fix the file permissions (Windows).** `ssh` refuses a key that other Windows users can read, and shows
   *"UNPROTECTED PRIVATE KEY FILE"*. These commands make the file readable by you only:

```powershell
$KEY = "$env:USERPROFILE\.ssh\civicai-key.pem"
icacls $KEY /reset
icacls $KEY /grant:r "${env:USERNAME}:(R)"
icacls $KEY /inheritance:r
icacls $KEY
```

The last command should list **only your own user name** with `(R)`. If other names appear, run the three
commands again.

## A6. Network settings and security group

In **Network settings** click **Edit**, and set:

* **Auto-assign public IP: Enable.** (Without a public IP you cannot reach the server.)
* **Firewall (security groups) > Create security group.** Name: `civicai-sg`. Description:
  `CivicAI API: SSH from my IP, port 8000 for the demo`.
* **Inbound security group rules** (exactly two):

| # | Type | Port | Source | Why |
| --- | --- | --- | --- | --- |
| 1 | SSH | 22 | **My IP** (the console fills in your current address) | Only your computer can log in. Anyone else who tries is blocked before they can even guess a key |
| 2 | Custom TCP | 8000 | **Anywhere-IPv4** (`0.0.0.0/0`); the console shows a warning, that is expected | Lets you, Postman and the grader call the API from any computer |

Remove any other rule the wizard adds (no HTTP 80, no HTTPS 443, no "All traffic"). The default outbound rule
(everything allowed) stays: the server needs it to download packages.

**Why an open port 8000 is a demo trade-off.** Anyone on the Internet who knows the address can call the API.
There is no login and no HTTPS (the traffic can be read on the way). That is acceptable for a short demo
because the API handles only public 311-style data, has no secrets, validates every input, and logs no
precise locations. It is *not* how a real service should be exposed. **After grading, close it:**
EC2 > Security Groups > `civicai-sg` > *Edit inbound rules* > delete the port 8000 rule > Save (or simply
terminate the instance).

*Two warnings about rule 1:* "My IP" is the address of the network you are on right now. If your Wi-Fi or
your campus network changes your address, ssh will time out: edit the rule and choose *My IP* again. Some
campus networks block port 22 altogether; try a phone hotspot.

## A7. Storage

Leave the default (**8 GiB, gp3**). It is enough and keeping the default avoids extra cost.
Leave *Delete on termination* as it is (yes).

## A8. Launch

Leave *Advanced details* alone. Check the **Summary** on the right: 1 instance, t3.micro, Ubuntu 22.04.
Click **Launch instance**. Open the instance list (**View all instances**). Wait 1 to 2 minutes until
*Instance state* is **Running** and *Status check* is **2/2 checks passed**.

## A9. Write down the public address

Select the instance. On the **Details** tab copy **Public IPv4 address** (for example `54.169.x.x`; not
the *Private* one). Your public address is fine to show in screenshots. **It changes every time you stop and
start the instance** (next section), so check it again at the start of every session.

## A10. A fixed address (Elastic IP): usually skip it

An *Elastic IP* is an address that stays yours when the instance is stopped and started. **It is not free
any more:** since 2024 AWS charges for every public IPv4 address (about 0.005 US dollars per hour), and an
Elastic IP that is allocated but not attached to a running instance is charged too. Older notes that say
"free while attached to a running instance" are out of date; check the pricing page.

*Recommendation:* skip it. Use the automatic address and copy the new one after each start (A9).
If you want one fixed address for the grading day: EC2 > **Elastic IPs** > *Allocate Elastic IP address* >
*Associate* it with `civicai-api`, and **release it right after grading** (*Disassociate*, then *Release*).

## A11. Stopping and finishing

* Done for today: select the instance > **Instance state > Stop instance**.
* Next session: **Instance state > Start instance**, then read the new public address (A9).
* After grading: **Instance state > Terminate (delete) instance**, delete the Elastic IP if you made one,
  and delete the budget if you like.

---

# PART B: commands

Everything in **B0 to B3** you type in **Windows PowerShell on your laptop**. From **B4** on you type on
the **server** (the prompt shows `ubuntu@ip-...:~$`). A line that starts with `#` is a comment: do not type it.

## B0. Set up your PowerShell window (once per window)

```powershell
cd "D:\Personal Folder\American University of Phnom Penh\Fall 2026\Cloud Machine Learning\ML-Final-Project"
$EC2_IP = "PASTE-THE-PUBLIC-IPV4-ADDRESS-HERE"
$KEY    = "$env:USERPROFILE\.ssh\civicai-key.pem"
```

(Use your own project path if it is different.)

## B1. Can you reach the server?

```powershell
ssh -i $KEY ubuntu@$EC2_IP
```

The first time, ssh asks *"Are you sure you want to continue connecting?"* Type `yes`. You should see the
Ubuntu welcome text and a prompt `ubuntu@ip-...:~$`. Type `exit` to come back. If it fails, go to the
troubleshooting table (section C).

## B2. Check the files on your laptop

```powershell
python -m pytest -q                    # 71 passed
python deploy/smoke_test.py            # needs the local server running; optional here
```

The copy script (next step) also checks that `models/model.joblib` is the file described in
`models/model_meta.json`.

## B3. Copy the files to the server

Always do the **dry run first**: it checks everything and only prints the commands.

```powershell
.\deploy\copy_to_ec2.ps1 -KeyPath $KEY -HostName $EC2_IP -User ubuntu -DryRun
.\deploy\copy_to_ec2.ps1 -KeyPath $KEY -HostName $EC2_IP -User ubuntu
```

(If PowerShell says running scripts is disabled: `powershell -ExecutionPolicy Bypass -File .\deploy\copy_to_ec2.ps1 -KeyPath $KEY -HostName $EC2_IP -User ubuntu`.)

**Exactly what is copied** (11 files, about 1.9 MB; the script refuses anything else):

| To (on the server, under `~/civicai/`) | Files | What they are |
| --- | --- | --- |
| `app/` | `__init__.py`, `main.py`, `predictor.py`, `schemas.py` | the API |
| `src/` | `__init__.py`, `features.py`, `preprocessing.py` | the three files the saved model needs to be loaded and run (the API imports `src.features` and `src.preprocessing`) |
| `models/` | `model.joblib`, `model_meta.json` | the model (1.8 MB) and its description |
| `./` | `requirements-api.txt` | the exact package versions |
| `deploy/` | `civicai.service` | the systemd file (step B9) |

**Never copied:** `data/`, `notebooks/`, `reports/`, `tests/`, `docs/`, the `.git` folder, `venv/`, `logs/`,
any `.pem` or `.key` file, any `.env` file. The script stops with an error if you try. The last thing it does
is compare the model's SHA-256 on the server with the one on your laptop.

*Manual alternative* (the same copy, one command at a time, from the project folder):

```powershell
ssh -i $KEY ubuntu@$EC2_IP "mkdir -p ~/civicai/app ~/civicai/src ~/civicai/models ~/civicai/deploy"
scp -i $KEY app/__init__.py app/main.py app/predictor.py app/schemas.py ubuntu@${EC2_IP}:~/civicai/app/
scp -i $KEY src/__init__.py src/features.py src/preprocessing.py ubuntu@${EC2_IP}:~/civicai/src/
scp -i $KEY models/model.joblib models/model_meta.json ubuntu@${EC2_IP}:~/civicai/models/
scp -i $KEY requirements-api.txt ubuntu@${EC2_IP}:~/civicai/
scp -i $KEY deploy/civicai.service ubuntu@${EC2_IP}:~/civicai/deploy/
```

## B4. Log in to the server

```powershell
ssh -i $KEY ubuntu@$EC2_IP
```

From here, every command is typed **on the server**.

## B5. Update Ubuntu and install Python's tools

```bash
sudo apt update
sudo apt -y upgrade
sudo apt -y install python3-venv python3-pip
python3 --version
```

The last line must print **Python 3.10.x**. If a purple screen titled *Daemons using outdated libraries*
appears, press Enter to accept the default.

*If the Python version is not 3.10* (for example you picked Ubuntu 24.04, which has 3.12): install 3.10 from a
well-known third-party package source, then use `python3.10` instead of `python3` in the next step:

```bash
sudo apt -y install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt -y install python3.10 python3.10-venv
python3.10 --version
```

## B6. Create the virtual environment and install the packages

A virtual environment is a private copy of Python for this project, so its packages do not mix with the
system's.

```bash
cd ~/civicai
python3 -m venv venv                  # on Ubuntu 24.04: python3.10 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install --no-cache-dir -r requirements-api.txt
```

This takes a few minutes (the XGBoost download is large). Then check that the versions are the ones the model
was built with:

```bash
./venv/bin/python -c "import sklearn, xgboost, pandas, numpy, joblib, fastapi, uvicorn, pydantic; print('scikit-learn', sklearn.__version__, '| xgboost', xgboost.__version__, '| pandas', pandas.__version__, '| numpy', numpy.__version__, '| joblib', joblib.__version__)"
python3 -c "import json; print(json.load(open('models/model_meta.json'))['versions'])"
```

The two lines must show the same scikit-learn (1.7.2), xgboost (3.2.0), pandas, numpy and joblib versions.

## B7. First test, in the foreground

```bash
cd ~/civicai
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

You should see lines like `Installed Python and library versions match`, `Model 3.0 loaded`, and
`Uvicorn running on http://0.0.0.0:8000`. Leave this window as it is. In a **second PowerShell window on your
laptop** (run B0 again there), test:

```powershell
Invoke-RestMethod "http://${EC2_IP}:8000/health"
```

You should get `status: ok`, `model_loaded: True`, `model_version: 3.0`. Also open
`http://<your-public-ip>:8000/docs` in a browser. Then go back to the server window and press **Ctrl+C**
to stop uvicorn. (A program started this way stops when you close the window: that is why B9 exists.)

## B8. Optional: look at the log without the service

Nothing to do here. The API writes `~/civicai/logs/predictions.jsonl` (no coordinates, ZIP codes or
descriptions are logged) and the console messages go to the journal (step B9).

## B9. Install the service so the API runs by itself

```bash
sudo cp ~/civicai/deploy/civicai.service /etc/systemd/system/civicai.service
sudo systemctl daemon-reload
sudo systemctl enable --now civicai
systemctl status civicai --no-pager
```

* `daemon-reload` makes systemd read the new file; `enable` makes it start at every boot; `--now` also starts it right now.
* The status must say **active (running)** and **enabled**.
* Look at the log: `journalctl -u civicai -n 40 --no-pager`. Follow it live (Ctrl+C to leave):
  `journalctl -u civicai -f`
* Check from the server itself: `curl -s localhost:8000/health`

If the unit's paths do not match your folders, edit it with `sudo nano /etc/systemd/system/civicai.service`
and run `sudo systemctl daemon-reload && sudo systemctl restart civicai`.

## B10. Run the smoke test from your laptop

```powershell
python deploy/smoke_test.py "http://${EC2_IP}:8000"
```

It prints a table of PASS/FAIL lines: health, model info, the five valid requests (each probability must
equal the saved value to 6 decimals), the three invalid requests (each must give 422 naming the field), and
the `/docs` page. The last line must say **RESULT: PASS**.

## B11. Reboot test (the service must come back by itself)

On the server:

```bash
sudo reboot
```

The connection closes. Wait about 60 to 90 seconds, log in again (B4) and run:

```bash
uptime -p
systemctl status civicai --no-pager
curl -s localhost:8000/health
```

`uptime -p` should say the server has been up for a minute or two, and the service should be **active
(running)** although you did not start it. (After a reboot the public address stays the same; only
stop/start changes it.)

## B12. Updating the model later

1. On the laptop: build or choose the new model, run `python -m pytest -q`.
2. *(optional safety copy)* on the server: `cp ~/civicai/models/model.joblib ~/civicai/models/model.joblib.bak`
3. On the laptop: run the copy script again (B3). It copies the new model and `model_meta.json` and checks the SHA-256.
4. If `requirements-api.txt` changed: on the server `cd ~/civicai && ./venv/bin/pip install -r requirements-api.txt`.
5. On the server: `sudo systemctl restart civicai`, then `journalctl -u civicai -n 20 --no-pager`.
6. On the laptop: `python deploy/smoke_test.py "http://${EC2_IP}:8000"`.

The API loads the model only when it starts, so **the restart is required**. To go back to the old model,
copy the `.bak` file back to `model.joblib` (and the old `model_meta.json`), then restart.

## B13. When you are finished

Stop the instance after each session; after grading terminate it and remove the port 8000 rule (A6, A11).

---

# C. Troubleshooting

| Symptom | Most likely cause | What to do |
| --- | --- | --- |
| `ssh: connect to host ... Connection timed out` (or `Operation timed out`) | The instance is stopped; or the **IP changed** after a stop/start; or the security group's port 22 rule is for an old "My IP"; or your network blocks port 22 (some campus networks) | In the console check *Instance state = Running* and copy the **current** public IPv4 (A9). Open `civicai-sg` > inbound rules > edit rule 1 > choose *My IP* again. Try a phone hotspot. As a test, console > *Connect* > *EC2 Instance Connect* opens a browser terminal |
| `Permission denied (publickey)` | Wrong user (it must be `ubuntu`); wrong key file (not the one chosen at launch); typo in `$KEY` | Use `ubuntu@...`. Check `Test-Path $KEY`. Run `ssh -v -i $KEY ubuntu@$EC2_IP` and read the last lines. If the key is lost, you cannot recover it: launch a new instance with a new key pair |
| `WARNING: UNPROTECTED PRIVATE KEY FILE!` / `bad permissions` | Other Windows users can read the .pem | Do the four `icacls` commands in A5 |
| `WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!` | A new instance got an address that an old one used | `ssh-keygen -R $EC2_IP`, then connect again |
| `scp`: `No such file or directory`, or `Permission denied` | The folders on the server were not created; wrong user | Use the script (it creates them), or run the `mkdir` command of the manual copy |
| `running scripts is disabled on this system` (PowerShell) | Execution policy | `powershell -ExecutionPolicy Bypass -File .\deploy\copy_to_ec2.ps1 ...` |
| `ModuleNotFoundError: No module named 'fastapi'` (or `pandas`, `xgboost`...) | The system Python was used, not the virtual environment; or `pip install` did not finish | Always use `./venv/bin/python`. Repeat `./venv/bin/pip install -r requirements-api.txt` and read its last lines |
| `ModuleNotFoundError: No module named 'src'` | uvicorn was started from the wrong folder, or `src/` was not copied | `cd ~/civicai` first; check `ls ~/civicai/src` shows `__init__.py features.py preprocessing.py` |
| `/health` shows `versions_match_model_meta: false`, or the journal says `VERSION MISMATCH` | Python is not 3.10, or the packages are not the pinned versions (scikit-learn **must be 1.7.2**) | Compare the two version lines in B6. Reinstall with `pip install -r requirements-api.txt`. On Ubuntu 24.04 use the Python 3.10 steps in B5 |
| `Killed` during `pip install` | Out of memory on the 1 GiB server | Add 1 GB of swap: `sudo fallocate -l 1G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`, then run pip again |
| The API answers **503** ("The prediction model is not loaded") and `/health` says `unavailable` | The model file is missing, in another place, damaged, or built for other library versions | `journalctl -u civicai -n 50 --no-pager` shows the reason. `ls -l ~/civicai/models` must list `model.joblib` and `model_meta.json`. Compare `sha256sum ~/civicai/models/model.joblib` with the SHA-256 in `model_meta.json`. If the path differs from `Environment=CIVICAI_MODEL_PATH=` in the unit, fix it, `sudo systemctl daemon-reload`, `sudo systemctl restart civicai` |
| `curl localhost:8000/health` works on the server but the laptop gets no answer | Security group has no rule for port 8000; you used `https://` instead of `http://`; a network firewall blocks port 8000 | Check rule 2 of `civicai-sg`. Use `http://<ip>:8000`. Try a phone hotspot |
| `[Errno 98] Address already in use` / `error while attempting to bind` | The foreground uvicorn from B7 (or another copy) is still using port 8000 | `sudo ss -ltnp \| grep 8000` shows who. Stop it (Ctrl+C in that window, or `sudo systemctl stop civicai`), then start only one |
| `systemctl status civicai` shows `failed`, `start request repeated too quickly`, or the service keeps restarting (flapping) | The program crashes at start; the unit allows 5 failures in 2 minutes and then stops trying | `journalctl -u civicai -n 50 --no-pager` shows the error (wrong path, missing module, model problem). Fix it, then `sudo systemctl reset-failed civicai` and `sudo systemctl restart civicai` |
| `status=203/EXEC` in the status output | `ExecStart` points to a file that does not exist (no virtual environment), **or** the unit file has Windows line endings | `ls ~/civicai/venv/bin/python` must exist (redo B6). Run `cat -A /etc/systemd/system/civicai.service \| head -3`: if the lines end in `^M`, the file is in Windows format; fix the local file as the copy script says, copy it again and repeat B9 |
| `apt` or `pip` cannot download anything | The instance has no public IP, or outbound traffic was removed | Instance details must show a public IPv4 address; outbound rule must allow all traffic |
| Valid requests answer 200 but the probabilities differ from the smoke test | Different model or different library versions on the server | Compare the SHA-256 (B3 does it); compare versions (B6) |
| Everything worked yesterday, now nothing answers | The instance was stopped and started: the **public IP changed** | Copy the new address (A9) and update `$EC2_IP` and the Postman variable `base_url` |
