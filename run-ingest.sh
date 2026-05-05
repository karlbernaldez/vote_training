#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:-}"

if [ -z "$BACKEND" ]; then
  echo "Usage: ./run-ingest.sh gcs|s3"
  exit 1
fi

if [ "$BACKEND" = "gcs" ]; then
  docker run --rm \
    --env-file ./.env \
    -v /home/vote/vote_training/Scripts/Homework/4/wavelab-489808-709da7d35394.json:/app/gcp-key.json:ro \
    -e STORAGE_BACKEND=gcs \
    -e GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json \
    vote-ingest:latest

elif [ "$BACKEND" = "s3" ]; then
  docker run --rm \
    --env-file ./.env \
    -e STORAGE_BACKEND=s3 \
    vote-ingest:latest

else
  echo "Invalid backend: $BACKEND"
  echo "Usage: ./run-ingest.sh gcs|s3"
  exit 1
fi