# ML Dataset Preparation

This package converts transform outputs into CNN-ready NumPy tensors and provides optional PyTorch training scripts.

## Inputs

The preparation scripts expect outputs from the transform pipeline:

- `station_wind` ML CSV files for station sequence datasets
- `gridded_wind` NetCDF files for grid CNN datasets

Prediction targets can be built from prepared tensors for smoke tests and baseline proxy experiments. Production supervised training should eventually use observed labels.

## Install ML dependencies

The ingest/pipeline Docker image does not install PyTorch by default. For ML training, install the optional requirements in your virtual environment:

```bash
pip install -r requirements-ml.txt
```

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

## Build target CSVs

Build default wind proxy targets from a prepared dataset:

```bash
python -m ml.build_targets \
  --dataset-dir data/ml/gridded_wind_conv3d \
  --output-csv data/ml/targets.csv
```

Default target columns:

```text
max_wind_speed_kph
mean_wind_speed_kph
p95_wind_speed_kph
strong_wind_event
```

Add final forecast-step targets for Conv3d or station Conv1d datasets:

```bash
python -m ml.build_targets \
  --dataset-dir data/ml/gridded_wind_conv3d \
  --output-csv data/ml/targets_final_step.csv \
  --targets max_wind_speed_kph mean_wind_speed_kph final_step_mean_wind_speed_kph final_step_max_wind_speed_kph strong_wind_event
```

Use a different threshold for the binary strong-wind event target:

```bash
python -m ml.build_targets \
  --dataset-dir data/ml/gridded_wind_conv3d \
  --output-csv data/ml/targets_50kph.csv \
  --strong-wind-threshold-kph 50
```

These generated targets are proxy targets derived from model input data. They are useful for checking the training loop and building baselines. For a real forecast model, replace or join them with observed labels.

## Pretrain a 3-layer Conv3D autoencoder

The autoencoder reconstructs `X.npy` and can learn compact spatial-temporal features without labels.

```bash
python -m ml.train_autoencoder \
  --dataset-dir data/ml/gridded_wind_conv3d \
  --output-dir data/ml/runs/autoencoder_smoke \
  --epochs 1 \
  --batch-size 1 \
  --device cpu
```

Outputs:

```text
autoencoder.pt
autoencoder_metrics.json
```

For longer training, increase `--epochs`. Use `--device cuda` only when PyTorch can access a GPU.

## Train a Conv3D predictor

Predictor training requires a target CSV with one row per sample in `X.npy`.

Train from scratch:

```bash
python -m ml.train_predictor \
  --dataset-dir data/ml/gridded_wind_conv3d \
  --target-csv data/ml/targets.csv \
  --target-columns max_wind_speed_kph \
  --output-dir data/ml/runs/predictor_smoke \
  --epochs 1 \
  --batch-size 1 \
  --device cpu
```

Train using the pretrained autoencoder encoder:

```bash
python -m ml.train_predictor \
  --dataset-dir data/ml/gridded_wind_conv3d \
  --target-csv data/ml/targets.csv \
  --target-columns max_wind_speed_kph \
  --pretrained-autoencoder data/ml/runs/autoencoder_smoke/autoencoder.pt \
  --output-dir data/ml/runs/predictor_pretrained_smoke \
  --epochs 1 \
  --batch-size 1 \
  --device cpu
```

Add `--freeze-encoder` if you want to train only the predictor head at first.

## Validation

Run pipeline-safe tests:

```bash
pytest -q
```

Run optional ML tests after installing PyTorch:

```bash
pip install -r requirements-ml.txt
pytest -q tests/test_cnn_training.py
```

Before merge/deployment, still validate the full operational path:

```bash
docker build -f Dockerfile.pipeline -t vote-ingest-pipeline:latest .
TRANSFORM_MODE=station_wind,gridded_wind ./run-pipeline.sh gcs
```

Do not change crontab as part of ML dataset preparation or training experiments.
