# Ingest Tests

These tests check that the ingest script still works when code changes are made.

They are focused on the ingest workflow in:

```text
Scripts/Homework/4/ingest.py
```

## What the tests cover

The tests make sure the script can:

- Build a correct `manifest.json` summary.
- Write ingest logs and manifests.
- Support both GCS and S3/Ceph storage backends.
- Parse a Ceph-style bucket URL, such as:

  ```text
  http://10.11.1.171:30080/raw-ingest
  ```

- Use either Ceph credentials or AWS-style credentials.
- Configure the S3 client with path-style addressing for Ceph.
- Skip the S3 `HeadObject` exists check when `S3_SKIP_EXISTS_CHECK=true`.
- Raise a helpful error when S3/Ceph blocks the exists check with `AccessDenied`.

## What the tests do not do

The tests do not contact real external systems.

They do not actually:

- Download data from NOAA.
- Upload files to GCS.
- Upload files to S3 or Ceph.
- Require real cloud credentials.

Instead, the tests use fake downloads, fake uploads, and mocked clients so they can run safely in GitHub Actions.

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

The GitHub Actions workflow runs the same test command automatically:

```bash
pytest -q
```

This helps catch regressions before the `ingest` branch is merged.

## Why these tests matter

The ingest branch added support for S3-compatible Ceph uploads while keeping the older GCS flow working.

These tests protect the important behavior:

- GCS should still work.
- S3/Ceph should work.
- Manifest and ingest log output should stay consistent.
- The known Ceph `HeadObject` permission issue should remain handled.
