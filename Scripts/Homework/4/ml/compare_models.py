from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    "mae",
    "rmse",
    "mape",
    "r2",
]


def load_metrics(path: Path):

    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None

    with open(path, "r") as f:
        return json.load(f)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        default="data/ml/benchmark",
    )

    parser.add_argument(
        "--lstm",
        default="data/ml/runs/lstm_v1/metrics.json",
    )

    parser.add_argument(
        "--gru",
        default="data/ml/evaluation/gru/metrics.json",
    )

    parser.add_argument(
        "--transformer",
        default="data/ml/evaluation/transformer/metrics.json",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = {
        "LSTM": load_metrics(Path(args.lstm)),
        "GRU": load_metrics(Path(args.gru)),
        "Transformer": load_metrics(
            Path(args.transformer)
        ),
    }

    rows = []

    for model_name, metrics in models.items():

        if metrics is None:
            continue

        row = {
            "model": model_name,
        }

        for metric in METRICS:
            row[metric] = metrics.get(
                metric,
                None,
            )

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No metrics files found."
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by="rmse",
        ascending=True,
    )

    df.insert(
        0,
        "rank",
        range(
            1,
            len(df) + 1,
        ),
    )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    csv_path = (
        output_dir
        / "leaderboard.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    # --------------------------------------------------
    # Save JSON
    # --------------------------------------------------

    json_path = (
        output_dir
        / "leaderboard.json"
    )

    with open(
        json_path,
        "w",
    ) as f:
        json.dump(
            df.to_dict(
                orient="records"
            ),
            f,
            indent=2,
        )

    # --------------------------------------------------
    # RMSE Chart
    # --------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        df["model"],
        df["rmse"],
    )

    plt.ylabel("RMSE")

    plt.title(
        "Model Comparison (Lower is Better)"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "leaderboard.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------
    # Console Output
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("MODEL LEADERBOARD")
    print("=" * 60)
    print()

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print(
        f"CSV  : {csv_path}"
    )
    print(
        f"JSON : {json_path}"
    )
    print(
        f"PNG  : {output_dir / 'leaderboard.png'}"
    )

    print()
    print(
        f"Winner: {df.iloc[0]['model']}"
    )


if __name__ == "__main__":
    main()