# ML Dataset Preparation

This package converts transform outputs into CNN-ready NumPy tensors.

## Inputs

The preparation scripts expect outputs from the transform pipeline:

- `station_wind` ML CSV files for station sequence datasets
- `gridded_wind` NetCDF files for grid CNN datasets

Targets are intentionally not included yet. Add targets only after the observed labels are defined and validated.

## Station dataset: Conv1d

Use station ML CSV files:

```bash
python -m ml.prepare_station_dataset \
  --input path/to/*station_wind_ml*.csv \
  --output-dir data/ml/station_wind_cnn1d
```

Output files:

```text
X.npy
metadata.csv
dataset_manifest.json
```

Tensor layout:

```text
N x C x T
```

Where:

- `N` = one station/run sample
- `C` = feature channels
- `T` = forecast steps

Default channels:

```text
u10_ms
v10_ms
wind_speed_kph
wind_dir_sin
wind_dir_cos
```

This layout is ready for PyTorch `Conv1d` as `batch x channels x sequence_length`.

## Gridded dataset: Conv2d

Use gridded wind NetCDF files:

```bash
python -m ml.prepare_grid_dataset \
  --input path/to/*gridded_wind*.nc \
  --output-dir data/ml/gridded_wind_conv2d \
  --layout conv2d
```

Tensor layout:

```text
N x C x H x W
```

Where each sample is one forecast hour.

## Gridded dataset: Conv3d

```bash
python -m ml.prepare_grid_dataset \
  --input path/to/*gridded_wind*.nc \
  --output-dir data/ml/gridded_wind_conv3d \
  --layout conv3d
```

Tensor layout:

```text
N x C x T x H x W
```

Where each sample is one full forecast sequence.

## Validation

Run:

```bash
pytest -q
```

Before merge/deployment, still validate the full operational path:

```bash
docker build -f Dockerfile.pipeline -t vote-ingest-pipeline:latest .
TRANSFORM_MODE=station_wind,gridded_wind ./run-pipeline.sh gcs
```

Do not change crontab as part of ML dataset preparation.
