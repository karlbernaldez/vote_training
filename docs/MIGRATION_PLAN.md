# Homework/4 Migration Plan

## Goal
Make pipelines/ the authoritative implementation and retire Scripts/Homework/4.

## Phase 1 - Shared Foundation
- Extract env_flag/env_value into pipelines/shared/config
- Extract sha256sum into pipelines/shared/hashing
- Extract parse_date/daterange into pipelines/shared/dates
- Centralize manifest generation
- Consolidate storage abstractions

## Phase 2 - GFS Ingest
- Port Homework/4 ingest workflow into pipelines/atmospheric/gfs
- Replace migration stub implementation
- Preserve GCS and S3 compatibility

## Phase 3 - Wind Transforms
- Create pipelines/shared/wind package
- Move dataset loading and feature engineering utilities
- Reuse across GFS and WW3

## Phase 4 - Validation
- Add parity tests between Homework/4 and pipelines implementations
- Verify outputs and manifests

## Phase 5 - Legacy Removal
- Freeze Homework/4
- Remove duplicated logic
- Delete legacy implementation after parity is confirmed
