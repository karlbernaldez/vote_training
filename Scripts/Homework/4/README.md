# WW3 GFS Ingestion Setup Guide.

This guide helps:
- get the project from GitHub to their local machine
- understand the basic Git commands they will use
- set up Python and project dependencies
- configure Google Cloud credentials
- run the WW3 GFS ingestion script

---

## 1. What this project does

This project downloads GFS GRIB2 input files for WW3 modeling and uploads them to a Google Cloud Storage bucket.

Current storage target:

```text
gs://<YOUR_BUCKET>/vote/bronze/GFS/YYYY/MM/DD/HH/FFF/<file>.grib2
```

Example:

```text
gs://vote3/WaveWatchIII/bronze/GFS/2024/04/13/00/003/2024041300_f003.grib2
```

---

## 2. What you need before starting

Install these first:

- Git
- Python 3.10 or newer
- VS Code or another code editor
- Access to the GitHub repository
- A Google Cloud service account JSON file with permission to upload to the bucket

### Check that Git and Python are installed

```bash
git --version
python --version
```

---

## 3. Git basics for beginners

### `git clone`
Use this the first time to download the repository to your computer.

```bash
git clone https://github.com/<ORG>/<REPO>.git
cd <REPO>
```

### `git fetch`
Use this to download the latest branch information and commits from GitHub **without changing your working files yet**.

```bash
git fetch
```

#### Why use `git fetch`?
It lets you safely see what changed on GitHub before updating your local branch. This is useful when:
- you want to inspect remote branches first
- you do not want to accidentally overwrite local work
- you want to switch to a branch that only exists on GitHub

Think of it as: **check what is new, but do not apply it yet.**

### `git checkout`
Use this to switch to a branch.

```bash
git checkout ingest
```

If the branch exists only on GitHub and not yet on your computer:

```bash
git fetch
git checkout ingest
```

or, if needed:

```bash
git checkout -b ingest origin/ingest
```

#### Why use `git checkout`?
Because the repository can have multiple branches for different features or tasks. You need to be on the correct branch before editing or running the latest version of the code.

Think of it as: **move into the version of the project you want to work on.**

### `git pull`
Use this to update your current branch with the latest changes from GitHub.

```bash
git pull origin ingest
```

#### Why use `git pull`?
Because your teammates may have pushed newer code. `git pull` updates your local branch so you run the latest version.

Think of it as: **bring the newest code into the branch you are already on.**

### Recommended beginner workflow

If this is your first time:

```bash
git clone https://github.com/<ORG>/<REPO>.git
cd <REPO>
git fetch
git checkout ingest
git pull origin ingest
```

If you already cloned the repo before:

```bash
git fetch
git checkout ingest
git pull origin ingest
```

### Simple explanation of the difference

- `git fetch` = download information about new commits and branches
- `git checkout` = switch to the branch you want to use
- `git pull` = update that branch with the latest code

A safe habit is:

```bash
git fetch
git checkout <branch>
git pull origin <branch>
```

---

## 4. Create a Python virtual environment

A virtual environment keeps project dependencies isolated from other Python projects on your machine.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation, you should see something like this in your terminal:

```text
(.venv)
```

---

## 5. Install project dependencies

Run this from the repository root:

```bash
pip install --upgrade pip
pip install -r Scripts/Homework/4/requirements.txt
```

---

## 6. Set up the service account JSON

You need a Google Cloud service account JSON file so the script can upload files to the bucket.

### Where to store it
Create a local folder named `keys` in the repository root and put the JSON file there.

Example:

```text
<REPO>/keys/ww3-sa.json
```

### Important
Do **not** commit this file to GitHub.

---

## 7. Create the `.env` file

Create this file:

```text
Scripts/Homework/4/.env
```

Example content:

```env
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your\repo\keys\ww3-sa.json
GCS_BUCKET_NAME=vote-ww3-lake-prod
GCS_PREFIX=vote

START_DATE=2024-04-13
END_DATE=2024-04-15
RUN_HOUR=00

FORECAST_STEP=3
FORECAST_MAX=72
```

### What these values mean

- `GOOGLE_APPLICATION_CREDENTIALS` = full path to your GCP service account JSON
- `GCS_BUCKET_NAME` = target bucket name
- `GCS_PREFIX` = root folder inside the bucket
- `START_DATE` = first date to download
- `END_DATE` = last date to download
- `RUN_HOUR` = model cycle hour, usually `00`
- `FORECAST_STEP` = forecast step interval, usually `3`
- `FORECAST_MAX` = max forecast hour, usually `72`

---

## 8. Run the ingestion script

Go to the script folder:

```bash
cd Scripts/Homework/4
```

Run the script:

```bash
python ingest.py
```

---

## 9. What the script will do

For each date from `START_DATE` to `END_DATE`, the script will:

1. build the GFS download URL
2. download forecast hours from `000` to `072` using the configured step
3. temporarily save each file locally
4. upload each file to the GCS bucket
5. create an `ingest_log.json` file for that run

---

## 10. Where the files go in GCS

Files are stored in this pattern:

```text
<bucket>/<prefix>/bronze/GFS/YYYY/MM/DD/HH/FFF/<filename>.grib2
```

Example:

```text
vote-ww3-lake-prod/vote/bronze/GFS/2024/04/13/00/003/2024041300_f003.grib2
```

---

## 11. Common Git commands your team will use

### See current status

```bash
git status
```

### Create a new branch for your work

```bash
git checkout -b feature/my-change
```

### Stage files

```bash
git add .
```

### Commit

```bash
git commit -m "feat(ingest): update GFS ingestion logic"
```

### Push your branch

```bash
git push origin feature/my-change
```

---

## 12. Common problems and fixes

### Problem: service account file not found

Error example:

```text
DefaultCredentialsError: File ... was not found
```

Fix:
- make sure the JSON file really exists
- use the full absolute path in `.env`
- check for typos in the filename

### Problem: 403 from NOMADS

Cause:
- the date may be too old for the operational NOMADS endpoint
- the request may be temporarily blocked or unavailable

Fix:
- try a recent date first
- use the correct source for historical backfills if needed

### Problem: Python modules not found

Fix:

```bash
pip install -r Scripts/Homework/4/requirements.txt
```

### Problem: `.env` values are not being read

Make sure the script includes:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 13. Important rules for the team

- Do not commit `.env`
- Do not commit service account JSON files
- Always activate the virtual environment before running the script
- Always pull the latest code before starting work
- Use `git status` often so you know what changed

---

## 14. Recommended daily workflow

```bash
git fetch
git checkout ingest
git pull origin ingest
```

Then activate the virtual environment and run the script.

### Windows PowerShell

```powershell
.venv\Scripts\Activate
python Scripts/Homework/4/ingest.py
```

### Mac/Linux

```bash
source .venv/bin/activate
python Scripts/Homework/4/ingest.py
```

If you are making code changes:

```bash
git fetch
git checkout ingest
git pull origin ingest
git checkout -b feature/my-change
```

Then after editing:

```bash
git add .
git commit -m "feat(ingest): describe your change"
git push origin feature/my-change
```

---

## 15. Summary

### First-time setup
1. Clone the repo
2. Fetch and checkout the correct branch
3. Create and activate the virtual environment
4. Install requirements
5. Add the service account JSON
6. Create `.env`
7. Run the script

### Regular usage
1. Fetch latest changes
2. Checkout the correct branch
3. Pull latest code
4. Activate `.venv`
5. Run the script
