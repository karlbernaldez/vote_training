#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-}"
IMAGE="${IMAGE:-vote-ingest-pipeline:latest}"
ENV_FILE="${ENV_FILE:-./.env}"
GCP_KEY_CONTAINER_PATH="/app/gcp-key.json"

if [ -z "$BACKEND" ]; then
  echo "Usage: $0 gcs|s3|both"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

run_s3() {
  docker run --rm \
    --env-file "$ENV_FILE" \
    -e STORAGE_BACKEND=s3 \
    "$IMAGE"
}

run_with_gcs_key() {
  local backend="$1"
  local gcp_key_path="${GCP_KEY_PATH:-}"

  if [ -z "$gcp_key_path" ]; then
    echo "GCP_KEY_PATH is required when BACKEND=$backend"
    echo "Example: GCP_KEY_PATH=/home/vote/vote_training/Scripts/Homework/4/wavelab-489808-709da7d35394.json $0 $backend"
    exit 1
  fi

  if [ ! -f "$gcp_key_path" ]; then
    echo "GCP key file not found: $gcp_key_path"
    exit 1
  fi

  docker run --rm \
    --env-file "$ENV_FILE" \
    -v "$gcp_key_path:$GCP_KEY_CONTAINER_PATH:ro" \
    -e STORAGE_BACKEND="$backend" \
    -e GOOGLE_APPLICATION_CREDENTIALS="$GCP_KEY_CONTAINER_PATH" \
    "$IMAGE"
}

case "$BACKEND" in
  gcs)
    run_with_gcs_key gcs
    ;;

  s3)
    run_s3
    ;;

  both)
    run_with_gcs_key both
    ;;

  *)
    echo "Invalid backend: $BACKEND"
    echo "Usage: $0 gcs|s3|both"
    exit 1
    ;;
esac
