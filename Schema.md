# Data Lake Schema

This schema describes the metadata model for ingesting GFS / WaveWatch output into object storage. The current ingest branch writes objects to a data lake path such as:

```text
s3://raw-ingest/WaveWatchIII/bronze/GFS/2026/04/26/00/000/2026042600_f000.grib2
```

General path template:

```text
{scheme}://{bucket}/{storage_prefix}/{lake_layer}/{source}/{yyyy}/{mm}/{dd}/{run_hour}/{forecast_hour}/{file_name}
```

Where:

- `scheme` is `gs` for GCS or `s3` for S3 / Ceph.
- `storage_prefix` is the top-level product or dataset prefix, for example `WaveWatchIII`.
- `lake_layer` is the lake zone, for example `bronze` for raw ingest output.
- `source` is the upstream model/source, for example `GFS`.
- Date partitions are split into `yyyy`, `mm`, `dd`, `run_hour`, and `forecast_hour`.

---

## Data lake path visualization

```mermaid
flowchart LR
    A[Storage Backend<br/>s3 or gs] --> B[Bucket<br/>raw-ingest or votewave]
    B --> C[Storage Prefix<br/>WaveWatchIII]
    C --> D[Lake Layer<br/>bronze]
    D --> E[Source<br/>GFS]
    E --> F[Year<br/>2026]
    F --> G[Month<br/>04]
    G --> H[Day<br/>26]
    H --> I[Run Hour<br/>00]
    I --> J[Forecast Hour<br/>000]
    J --> K[Object<br/>2026042600_f000.grib2]
```

Example expanded path:

```text
s3://raw-ingest
  /WaveWatchIII
    /bronze
      /GFS
        /2026
          /04
            /26
              /00
                /000
                  /2026042600_f000.grib2
```

---

## Metadata relationship visualization

```mermaid
erDiagram
    AUTHOR ||--o{ DATASET : owns
    SOURCE_MODEL ||--o{ DATASET : defines
    DOMAIN ||--o{ DATASET : bounds
    DATASET ||--o{ DATASET_VARIABLE : contains
    VARIABLE ||--o{ DATASET_VARIABLE : describes
    SOURCE_MODEL ||--o{ VARIABLE_MAP : maps
    VARIABLE ||--o{ VARIABLE_MAP : normalizes
    DATASET ||--o{ INGEST_RUN : produces
    STORAGE_BACKEND ||--o{ INGEST_RUN : stores
    INGEST_RUN ||--o{ LAKE_OBJECT : writes
    DATASET ||--o{ LAKE_OBJECT : catalogs
    STORAGE_BACKEND ||--o{ LAKE_OBJECT : locates
    LAKE_OBJECT ||--|| LAKE_PARTITION : partitions

    AUTHOR {
        int id PK
        string first_name
        string last_name
        string email
    }

    SOURCE_MODEL {
        int id PK
        string source
        string model
        string provider
        string base_url
    }

    DOMAIN {
        int id PK
        string name
        decimal min_lat
        decimal max_lat
        decimal min_lon
        decimal max_lon
    }

    DATASET {
        int id PK
        int source_model_id FK
        int domain_id FK
        int author_id FK
        string name
        string format
        string type
        string default_storage_prefix
        string status
    }

    STORAGE_BACKEND {
        int id PK
        string backend
        string scheme
        string bucket
        string endpoint_url
        string addressing_style
        string signature_version
        string status
    }

    INGEST_RUN {
        int id PK
        int dataset_id FK
        int storage_backend_id FK
        date run_date
        string run_hour
        string run_time
        int forecast_step
        int forecast_max
        int record_count
        int downloaded_count
        int skipped_count
        int failed_count
        string manifest_uri
        string ingest_log_uri
        string status
    }

    LAKE_OBJECT {
        int id PK
        int dataset_id FK
        int storage_backend_id FK
        int ingest_run_id FK
        string lake_layer
        string storage_prefix
        string object_path
        string storage_uri
        string provider_uri
        string object_kind
        string file_name
        string format
        int file_size_bytes
        string checksum_sha256
        string status
    }

    LAKE_PARTITION {
        int id PK
        int lake_object_id FK
        string source
        string year
        string month
        string day
        string run_hour
        string run_time
        string forecast_hour
        int forecast_step
        int forecast_max
    }

    VARIABLE {
        int id PK
        string code
        string name
        string unit
        string value_type
    }

    DATASET_VARIABLE {
        int id PK
        int dataset_id FK
        int variable_id FK
    }

    VARIABLE_MAP {
        int id PK
        int source_model_id FK
        int variable_id FK
        string provider_code
        string provider_name
        string level
    }
```

---

## Dataset

Represents a logical dataset or model product, independent of a single physical object.

- `id` — primary key
- `source_model_id` — FK -> `SourceModel.id`
- `name` — string, for example `WaveWatchIII GFS wind ingest`
- `description` — string
- `domain_id` — FK -> `Domain.id`
- `format` — enum/string: `grib2`, `grib`, `netcdf`, `json`, `tif`
- `type` — enum/string: `gridded`, `spectra`
- `default_storage_prefix` — string, for example `WaveWatchIII`
- `status` — enum/string: `active`, `deprecated`, `failed`, `archived`
- `author_id` — FK -> `Author.id`
- `created_at` — timestamp
- `updated_at` — timestamp

---

## SourceModel

Represents an upstream data source and model family.

- `id` — primary key
- `source` — string, for example `GFS`
- `model` — string, for example `WW3`
- `provider` — string, for example `NOAA/NCEP`
- `base_url` — string
- `description` — string

---

## StorageBackend

Represents a storage system where lake objects are written.

- `id` — primary key
- `backend` — enum/string: `gcs`, `s3`
- `scheme` — enum/string: `gs`, `s3`
- `bucket` — string, for example `raw-ingest`
- `endpoint_url` — string / nullable, for example `http://10.11.1.171:30080` for Ceph
- `addressing_style` — enum/string / nullable: `path`, `virtual`
- `signature_version` — string / nullable, for example `s3v4`
- `region` — string / nullable
- `status` — enum/string: `active`, `disabled`

Do not store access keys or secrets in this table. Credentials should live in the runtime secret manager or environment variables.

---

## LakeObject

Represents one physical object in the data lake, such as a GRIB2 file, ingest log, or manifest.

- `id` — primary key
- `dataset_id` — FK -> `Dataset.id`
- `storage_backend_id` — FK -> `StorageBackend.id`
- `ingest_run_id` — FK -> `IngestRun.id`
- `lake_layer` — enum/string: `bronze`, `silver`, `gold`
- `storage_prefix` — string, for example `WaveWatchIII`
- `object_path` — string, for example `WaveWatchIII/bronze/GFS/2026/04/26/00/000/2026042600_f000.grib2`
- `storage_uri` — string, for example `s3://raw-ingest/WaveWatchIII/bronze/GFS/2026/04/26/00/000/2026042600_f000.grib2`
- `provider_uri` — string / nullable, backward-compatible provider-specific URI such as `gcs_uri` or `s3_uri`
- `object_kind` — enum/string: `data_file`, `ingest_log`, `manifest`, `metadata`
- `file_name` — string
- `format` — enum/string: `grib2`, `grib`, `netcdf`, `json`, `tif`
- `file_size_bytes` — integer / nullable
- `checksum_sha256` — string / nullable
- `status` — enum/string: `downloaded_and_uploaded`, `skipped_already_exists`, `failed`
- `error` — string / nullable
- `created_at` — timestamp

---

## LakePartition

Represents the partition values encoded in a lake object path.

- `id` — primary key
- `lake_object_id` — FK -> `LakeObject.id`
- `source` — string, for example `GFS`
- `year` — string, `YYYY`
- `month` — string, `MM`
- `day` — string, `DD`
- `run_hour` — string, `HH`
- `run_time` — string, `YYYYMMDDHH`
- `forecast_hour` — string, `000`, `003`, `006`, etc.
- `forecast_step` — integer / nullable
- `forecast_max` — integer / nullable

Recommended unique constraint:

- (`source`, `run_time`, `forecast_hour`, `lake_object_id`)

---

## IngestRun

Represents one execution of the ingest process for a date/run-hour range.

- `id` — primary key
- `dataset_id` — FK -> `Dataset.id`
- `storage_backend_id` — FK -> `StorageBackend.id`
- `run_date` — date
- `run_hour` — string, `HH`
- `run_time` — string, `YYYYMMDDHH`
- `forecast_step` — integer
- `forecast_max` — integer
- `record_count` — integer
- `downloaded_count` — integer
- `skipped_count` — integer
- `failed_count` — integer
- `manifest_uri` — string / nullable
- `ingest_log_uri` — string / nullable
- `status` — enum/string: `running`, `completed`, `completed_with_failures`, `failed`
- `started_at` — timestamp / nullable
- `finished_at` — timestamp / nullable
- `generated_at` — timestamp / nullable

---

## Domain

Represents a spatial domain for a dataset.

- `id` — primary key
- `name` — string
- `min_lat` — decimal / nullable
- `max_lat` — decimal / nullable
- `min_lon` — decimal / nullable
- `max_lon` — decimal / nullable
- `elevation` — string / nullable
- `range` — array/object / nullable

---

## Variable

Represents a normalized scientific variable.

- `id` — primary key
- `code` — string
- `name` — string
- `unit` — string
- `value_type` — string

---

## DatasetVariable

Maps datasets to the variables they contain.

- `id` — primary key
- `dataset_id` — FK -> `Dataset.id`
- `variable_id` — FK -> `Variable.id`

Recommended unique constraint:

- (`dataset_id`, `variable_id`)

---

## VariableMap

Maps upstream model/provider variable codes to normalized variables.

- `id` — primary key
- `source_model_id` — FK -> `SourceModel.id`
- `variable_id` — FK -> `Variable.id`
- `provider_code` — string, for example `UGRD` or `VGRD`
- `provider_name` — string / nullable
- `level` — string / nullable, for example `10 m above ground`

---

## Author

Represents a person or system that created or owns the dataset definition.

- `id` — primary key
- `first_name` — string
- `last_name` — string
- `email` — string / nullable

---

## Example records

### StorageBackend

```json
{
  "backend": "s3",
  "scheme": "s3",
  "bucket": "raw-ingest",
  "endpoint_url": "http://10.11.1.171:30080",
  "addressing_style": "path",
  "signature_version": "s3v4",
  "status": "active"
}
```

### LakeObject

```json
{
  "lake_layer": "bronze",
  "storage_prefix": "WaveWatchIII",
  "object_path": "WaveWatchIII/bronze/GFS/2026/04/26/00/000/2026042600_f000.grib2",
  "storage_uri": "s3://raw-ingest/WaveWatchIII/bronze/GFS/2026/04/26/00/000/2026042600_f000.grib2",
  "object_kind": "data_file",
  "file_name": "2026042600_f000.grib2",
  "format": "grib2",
  "status": "downloaded_and_uploaded"
}
```
