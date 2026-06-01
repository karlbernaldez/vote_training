from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def flatten(arr):
    return arr.reshape(-1)


def actual_vs_predicted(
    preds,
    targets,
    output_path,
):
    plt.figure(figsize=(8, 8))

    plt.scatter(
        targets,
        preds,
        alpha=0.7,
    )

    mn = min(
        targets.min(),
        preds.min(),
    )

    mx = max(
        targets.max(),
        preds.max(),
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        "--",
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(
        "Actual vs Predicted"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def residual_histogram(
    residuals,
    output_path,
):
    plt.figure(figsize=(8, 5))

    plt.hist(
        residuals,
        bins=20,
    )

    plt.xlabel("Residual")
    plt.ylabel("Count")

    plt.title(
        "Residual Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def residual_scatter(
    preds,
    residuals,
    output_path,
):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        preds,
        residuals,
        alpha=0.7,
    )

    plt.axhline(
        y=0,
        linestyle="--",
    )

    plt.xlabel("Prediction")
    plt.ylabel("Residual")

    plt.title(
        "Residual Scatter"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


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
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = flatten(
        np.load(args.predictions)
    )

    targets = flatten(
        np.load(args.targets)
    )

    residuals = (
        targets - predictions
    )

    actual_vs_predicted(
        predictions,
        targets,
        output_dir
        / "actual_vs_predicted.png",
    )

    residual_histogram(
        residuals,
        output_dir
        / "residual_histogram.png",
    )

    residual_scatter(
        predictions,
        residuals,
        output_dir
        / "residual_scatter.png",
    )

    print(
        f"Saved plots to {output_dir}"
    )


if __name__ == "__main__":
    main()