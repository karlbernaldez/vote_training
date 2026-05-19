# Ingest Tests

These tests are safety checks for the ingest code in:

```text
Scripts/Homework/4/ingest.py
```

Simple version: **the tests make sure the ingest script can create logs and manifests, work with GCS and Ceph, handle credentials, and keep the known Ceph upload workaround working.**

## What the tests check

### 1. The script can make a manifest

The tests check that `manifest.json` correctly counts:

- downloaded files
- skipped files
- failed files

This matters because the manifest tells us what happened during an ingest run.

### 2. The script works with both storage systems

The tests check both supported storage backends:

- GCS paths like `gs://...`
- S3/Ceph paths like `s3://...`

This helps make sure the new S3/Ceph support does not break the older GCS flow.

### 3. The script can skip files that already exist

If a file is already in the bucket, the tests make sure the script does not download or upload it again.

The expected status is:

```text
SKIPPED_ALREADY_EXISTS
```

### 4. The script understands the Ceph bucket URL

The tests check that this value:

```text
http://10.11.1.171:30080/raw-ingest
```

is parsed as:

```text
endpoint = http://10.11.1.171:30080
bucket   = raw-ingest
```

### 5. The script handles credentials correctly

The tests check that the script accepts either:

```text
CEPH_ACCESS_KEY
CEPH_SECRET_KEY
```

or:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

### 6. The script handles the Ceph permission issue

Some Ceph setups allow uploads but block the `HeadObject` exists check.

The tests make sure this workaround works:

```bash
S3_SKIP_EXISTS_CHECK=true
```

They also check that when Ceph blocks the exists check with `AccessDenied`, the script gives a helpful error.

## What the tests do not do

The tests do not use real external services.

They do not actually:

- download data from NOAA
- upload files to GCS
- upload files to S3 or Ceph
- require real cloud credentials

Instead, they use fake downloads, fake uploads, and mocked clients so they can run safely in GitHub Actions.

## Run the tests locally

From the repository root, install dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
pytest -q
```

## GitHub Actions

GitHub Actions runs the same command automatically:

```bash
pytest -q
```

This helps catch problems before the `ingest` branch is merged.
