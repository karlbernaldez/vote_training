#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-}"
IMAGE="${IMAGE:-vote-ingest-pipeline:latest}"
ENV_FILE="${ENV_FILE:-./.env}"
GCP_KEY_CONTAINER_PATH="/app/gcp-key.json"
STARTED_AT="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
}

finish() {
  local exit_code="$?"
  local finished_at
  finished_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  if [ "$exit_code" -eq 0 ]; then
    log "Pipeline runner finished successfully backend=$BACKEND image=$IMAGE started_at=$STARTED_AT finished_at=$finished_at"
  else
    log "Pipeline runner failed exit_code=$exit_code backend=$BACKEND image=$IMAGE started_at=$STARTED_AT finished_at=$finished_at"
  fi
  exit "$exit_code"
}
trap finish EXIT

if [ -z "$BACKEND" ]; then
  log "Usage: $0 gcs|s3|both"
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  log "Missing env file: $ENV_FILE"
  exit 1
fi

log "Pipeline runner started backend=$BACKEND image=$IMAGE env_file=$ENV_FILE"

run_s3() {
  log "Starting Docker pipeline backend=s3"
  docker run --rm \
    --env-file "$ENV_FILE" \
    -e STORAGE_BACKEND=s3 \
    "$IMAGE"
  log "Docker pipeline completed backend=s3"
}

run_with_gcs_key() {
  local backend="$1"
  local gcp_key_path="${GCP_KEY_PATH:-}"

  if [ -z "$gcp_key_path" ]; then
    log "GCP_KEY_PATH is required when BACKEND=$backend"
    log "Example: GCP_KEY_PATH=/home/vote/vote_training/Scripts/Homework/4/wavelab-489808-709da7d35394.json $0 $backend"
    exit 1
  fi

  if [ ! -f "$gcp_key_path" ]; then
    log "GCP key file not found: $gcp_key_path"
    exit 1
  fi

  log "Starting Docker pipeline backend=$backend gcp_key_container_path=$GCP_KEY_CONTAINER_PATH"
  docker run --rm \
    --env-file "$ENV_FILE" \
    -v "$gcp_key_path:$GCP_KEY_CONTAINER_PATH:ro" \
    -e STORAGE_BACKEND="$backend" \
    -e GOOGLE_APPLICATION_CREDENTIALS="$GCP_KEY_CONTAINER_PATH" \
    "$IMAGE"
  log "Docker pipeline completed backend=$backend"
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
    log "Invalid backend: $BACKEND"
    log "Usage: $0 gcs|s3|both"
    exit 1
    ;;
esac
