from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(predictions, targets):

    mae = np.mean(
        np.abs(predictions - targets)
    )

    mse = np.mean(
        (predictions - targets) ** 2
    )

    rmse = np.sqrt(mse)

    mape = np.mean(
        np.abs(
            (targets - predictions)
            / (targets + 1e-6)
        )
    ) * 100

    ss_res = np.sum(
        (targets - predictions) ** 2
    )

    ss_tot = np.sum(
        (targets - np.mean(targets)) ** 2
    )

    r2 = 1.0 - (
        ss_res / (ss_tot + 1e-12)
    )

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
    )

    parser.add_argument(
        "--targets",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    predictions = np.load(
        args.predictions
    )

    targets = np.load(
        args.targets
    )

    metrics = evaluate(
        predictions,
        targets,
    )

    output = Path(args.output)

    with open(output, "w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print("\nMetrics")
    print("-" * 40)

    for k, v in metrics.items():
        print(
            f"{k:<10} {v:.6f}"
        )


if __name__ == "__main__":
    main()