#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-gcs}"
DAYS="${DAYS:-10}"
RUN_HOUR="${RUN_HOUR:-00}"
TRANSFORM_MODE="${TRANSFORM_MODE:-station_wind,gridded_wind}"
IMAGE="${IMAGE:-vote-ingest-pipeline:latest}"
ENV_FILE="${ENV_FILE:-./.env}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-./run-pipeline.sh}"
START_OFFSET_DAYS="${START_OFFSET_DAYS:-1}"

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
}

usage() {
  cat <<'EOF'
Usage:
  DAYS=10 GCP_KEY_PATH=/path/to/key.json ./backfill-pipeline.sh gcs

Environment variables:
  DAYS                 Number of dates to backfill. Default: 10.
  START_OFFSET_DAYS    Days before today to start from. Default: 1, meaning yesterday.
  RUN_HOUR             GFS run hour. Default: 00.
  TRANSFORM_MODE       Transform modes. Default: station_wind,gridded_wind.
  IMAGE                Docker image. Default: vote-ingest-pipeline:latest.
  ENV_FILE             Env file passed to run-pipeline.sh. Default: ./.env.
  PIPELINE_SCRIPT      Pipeline runner. Default: ./run-pipeline.sh.
  GCP_KEY_PATH         Required for gcs backend by run-pipeline.sh.

This script does not edit crontab. It runs the existing Docker pipeline once per date.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if ! [[ "$DAYS" =~ ^[0-9]+$ ]] || [ "$DAYS" -le 0 ]; then
  log "DAYS must be a positive integer. Got: $DAYS"
  exit 1
fi

if ! [[ "$START_OFFSET_DAYS" =~ ^[0-9]+$ ]]; then
  log "START_OFFSET_DAYS must be a non-negative integer. Got: $START_OFFSET_DAYS"
  exit 1
fi

if [ ! -x "$PIPELINE_SCRIPT" ]; then
  log "Pipeline script is missing or not executable: $PIPELINE_SCRIPT"
  exit 1
fi

log "Backfill started backend=$BACKEND days=$DAYS start_offset_days=$START_OFFSET_DAYS run_hour=$RUN_HOUR transform_mode=$TRANSFORM_MODE image=$IMAGE"

for offset in $(seq "$START_OFFSET_DAYS" $((START_OFFSET_DAYS + DAYS - 1))); do
  run_date="$(date -u -d "${offset} days ago" +%Y-%m-%d)"
  log "Backfill date started run_date=$run_date run_hour=$RUN_HOUR backend=$BACKEND"

  RUN_DATE="$run_date" \
  RUN_HOUR="$RUN_HOUR" \
  TRANSFORM_MODE="$TRANSFORM_MODE" \
  IMAGE="$IMAGE" \
  ENV_FILE="$ENV_FILE" \
  "$PIPELINE_SCRIPT" "$BACKEND"

  log "Backfill date finished run_date=$run_date run_hour=$RUN_HOUR backend=$BACKEND"
done

log "Backfill finished successfully backend=$BACKEND days=$DAYS"
