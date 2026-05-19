#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-}"
IMAGE="${IMAGE:-vote-ingest:latest}"
ENV_FILE="${ENV_FILE:-./.env}"

if [ -z "$BACKEND" ]; then
  echo "Usage: $0 gcs|s3"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

case "$BACKEND" in
  gcs)
    GCP_KEY_PATH="${GCP_KEY_PATH:-}"
    if [ -z "$GCP_KEY_PATH" ]; then
      echo "GCP_KEY_PATH is required when BACKEND=gcs"
      exit 1
    fi
    if [ ! -f "$GCP_KEY_PATH" ]; then
      echo "GCP key file not found: $GCP_KEY_PATH"
      exit 1
    fi

    docker run --rm \
      --env-file "$ENV_FILE" \
      -v "$GCP_KEY_PATH:/app/gcp-key.json:ro" \
      -e STORAGE_BACKEND=gcs \
      -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json \
      "$IMAGE"
    ;;

  s3)
    docker run --rm \
      --env-file "$ENV_FILE" \
      -e STORAGE_BACKEND=s3 \
      "$IMAGE"
    ;;

  *)
    echo "Invalid backend: $BACKEND"
    echo "Usage: $0 gcs|s3"
    exit 1
    ;;
esac