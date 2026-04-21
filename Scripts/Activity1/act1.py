from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

@dataclass
class Config:
    file_path: str = "../Dataset/swh_nrt_c2_l3_2025010100_2026010100.csv"
    output_dir: str = "plots/2"
    max_pairplot_samples: int = 5_000
    heatmap_samples: int = 2_000
    outlier_trim: float = 0.01          # drop bottom/top 1%
    random_state: int = 42
    target_col: str = "value"
    quality_col: str = "value_qc"
    quality_flag: int = 0               # keep only "good" records
    metadata_cols: list[str] = field(default_factory=lambda: [
        "variable", "platform_id", "platform_type",
        "institution", "doi", "product_doi", "value_qc",
    ])
    critical_cols: list[str] = field(default_factory=lambda: [
        "value", "latitude", "longitude",
    ])


# ──────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────

def _apply_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="viridis", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def _save(fig_or_path: str, output_dir: str) -> None:
    """Save the current figure and close it."""
    path = Path(output_dir) / f"{fig_or_path}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {path}")


# ──────────────────────────────────────────────
# PIPELINE STAGES
# ──────────────────────────────────────────────

def load(cfg: Config) -> pd.DataFrame:
    print("\n── 1. LOAD ──────────────────────────────")
    df = pd.read_csv(cfg.file_path)
    print(df.info())
    print(df.head())

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df


def report_missing(df: pd.DataFrame, cfg: Config) -> None:
    print("\n── 2. MISSING DATA ──────────────────────")
    missing_pct = df.isnull().mean().mul(100).sort_values(ascending=False)
    print(missing_pct.to_string())

    plt.figure(figsize=(10, 4))
    missing_pct.plot(kind="bar")
    plt.title("Missing Data %")
    plt.xticks(rotation=45)
    _save("missing_data", cfg.output_dir)


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("\n── 3. CLEAN ─────────────────────────────")
    print(f"  Initial shape: {df.shape}")

    # 1. Drop fully-empty columns
    full_missing = df.columns[df.isnull().all()].tolist()
    if full_missing:
        print(f"  Dropping fully-missing cols: {full_missing}")
        df = df.drop(columns=full_missing)

    # 2. Quality filter
    if cfg.quality_col in df.columns:
        print(f"  Filtering {cfg.quality_col} == {cfg.quality_flag}")
        df = df[df[cfg.quality_col] == cfg.quality_flag]

    # 3. Drop metadata / admin columns
    drop_cols = [c for c in cfg.metadata_cols if c in df.columns]
    if drop_cols:
        print(f"  Dropping metadata: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # 4. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df):,} duplicate rows")

    # 5. Drop rows missing critical columns
    critical = [c for c in cfg.critical_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"  Dropped {before - len(df):,} rows with missing critical values")

    # 6. Trim extreme outliers in target column
    if cfg.target_col in df.columns:
        lo = df[cfg.target_col].quantile(cfg.outlier_trim)
        hi = df[cfg.target_col].quantile(1 - cfg.outlier_trim)
        before = len(df)
        df = df[df[cfg.target_col].between(lo, hi)]
        print(f"  Removed {before - len(df):,} extreme outliers ({cfg.outlier_trim*100:.0f}% tails)")

    print(f"  Final shape:   {df.shape}")
    return df


def _drop_low_variance(df: pd.DataFrame, num_cols: pd.Index, threshold: float = 1e-6) -> pd.Index:
    """Return only columns whose normalised variance exceeds threshold."""
    variances = df[num_cols].var()
    ranges = df[num_cols].max() - df[num_cols].min()
    norm_var = variances / (ranges ** 2).replace(0, np.nan)
    useful = norm_var[norm_var > threshold].index
    dropped = set(num_cols) - set(useful)
    if dropped:
        print(f"  Skipping low-variance columns in plots: {sorted(dropped)}")
    return useful


def analyse_features(df: pd.DataFrame, cfg: Config) -> None:
    print("\n── 4. FEATURE ANALYSIS ──────────────────")
    num_cols = df.select_dtypes(include=np.number).columns
    print(df[num_cols].describe())

    if len(num_cols) < 2:
        return

    # Drop near-constant columns before plotting
    plot_cols = _drop_low_variance(df, num_cols)
    if len(plot_cols) < 2:
        print("  Not enough variance in numeric columns to plot correlations.")
        return

    corr = df[plot_cols].corr()

    # — Correlation heatmap (only meaningful pairs) —
    n = len(plot_cols)
    fig_size = max(6, n * 1.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=1.5, linecolor="white",
        square=True, ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation", fontsize=15, pad=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=40, labelsize=11)
    ax.tick_params(axis="y", rotation=0,  labelsize=11)
    fig.tight_layout()
    _save("pearson_heatmap", cfg.output_dir)

    # — Pairplot (top correlated features) —
    _pairplot(df, corr, plot_cols, cfg)


def _pairplot(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    plot_cols: pd.Index,
    cfg: Config,
) -> None:
    target = cfg.target_col

    # Select columns: target + up to 3 most correlated useful features
    if target in df.columns and target in plot_cols:
        top = (
            corr[target].drop(target, errors="ignore")
            .abs().nlargest(3).index.tolist()
        )
        cols = [target] + [c for c in top if c in plot_cols]
    else:
        cols = list(plot_cols[:4])

    if len(cols) < 2:
        print("  Skipping pairplot — fewer than 2 plottable columns.")
        return

    sample = df[cols].dropna()
    if len(sample) > cfg.max_pairplot_samples:
        sample = sample.sample(cfg.max_pairplot_samples, random_state=cfg.random_state)

    # Build continuous color array from target column
    colors = None
    norm = None
    if target in sample.columns:
        norm = plt.Normalize(sample[target].min(), sample[target].max())
        colors = plt.cm.plasma(norm(sample[target].values))

    n_cols = len(cols)
    cell_size = 2.8
    g = sns.PairGrid(
        sample, corner=True,
        height=cell_size, aspect=1.0,
        despine=True,
    )

    # Scatter: drop seaborn's injected `color` kwarg so it doesn't clash with `c`
    def _scatter(x, y, **kwargs):
        kwargs.pop("color", None)
        plt.gca().scatter(
            x, y,
            c=colors, cmap="plasma",
            s=8, alpha=0.55, edgecolor="none", linewidths=0,
        )

    g.map_lower(_scatter)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.55, linewidth=1.2, warn_singular=False)

    # Style each active axis
    for ax_row in g.axes:
        for ax in ax_row:
            if ax is not None:
                ax.tick_params(labelsize=9)
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                ax.set_ylabel(ax.get_ylabel(), fontsize=10)

    # Add colorbar anchored to last (bottom-right) diagonal cell
    if norm is not None:
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        last_diag = g.axes[n_cols - 1][n_cols - 1]
        if last_diag is not None:
            cbar = g.figure.colorbar(sm, ax=last_diag, fraction=0.08, pad=0.04, shrink=0.8)
            cbar.set_label(target.capitalize(), fontsize=10)
            cbar.ax.tick_params(labelsize=9)

    g.figure.suptitle("Feature pairplot (plasma = wave height)", y=1.01, fontsize=13, fontweight="bold")
    g.figure.tight_layout()
    path = Path(cfg.output_dir) / "pairplot_top_features.png"
    g.figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def analyse_distributions(df: pd.DataFrame, cfg: Config) -> None:
    print("\n── 5. DISTRIBUTIONS ─────────────────────")
    target = cfg.target_col

    # Value histogram — filled KDE overlay, bold median line
    if target in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            df[target], bins=50, kde=True, ax=ax,
            color="#4C72B0", alpha=0.55,
            line_kws={"linewidth": 2.2, "color": "#1a3a6b"},
        )
        median_val = df[target].median()
        ax.axvline(median_val, color="#e05c2a", linewidth=1.8, linestyle="--", label=f"Median {median_val:.2f}")
        ax.set_title(f"Distribution of {target}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(target.capitalize(), fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        sns.despine(fig=fig)
        fig.tight_layout()
        _save("value_distribution", cfg.output_dir)

    # Hourly mean — line + confidence band
    if "time" in df.columns and target in df.columns:
        df = df.copy()
        df["hour"] = df["time"].dt.hour
        hourly = df.groupby("hour")[target].agg(["mean", "std", "count"]).reset_index()
        hourly["se"] = hourly["std"] / np.sqrt(hourly["count"])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(
            hourly["hour"],
            hourly["mean"] - hourly["se"],
            hourly["mean"] + hourly["se"],
            alpha=0.25, color="#4C72B0", label="±1 SE",
        )
        ax.plot(hourly["hour"], hourly["mean"], marker="o", markersize=6,
                linewidth=2, color="#1a3a6b", label="Mean")
        ax.set_title(f"Mean {target} by hour (UTC)", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Hour of day", fontsize=11)
        ax.set_ylabel(f"Mean {target}", fontsize=11)
        ax.set_xticks(range(0, 24, 2))
        ax.legend(fontsize=10)
        sns.despine(fig=fig)
        fig.tight_layout()
        _save("value_by_hour", cfg.output_dir)


def analyse_spatial(df: pd.DataFrame, cfg: Config) -> None:
    print("\n── 6. SPATIAL ───────────────────────────")
    spatial_cols = {"latitude", "longitude"}

    if not spatial_cols.issubset(df.columns):
        print("  Skipped — lat/lon columns not found.")
        return

    # Scatter map — sized figure, better colormap, land-like background
    if cfg.target_col in df.columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor("#d6e8f5")
        ax.set_facecolor("#d6e8f5")

        sc = ax.scatter(
            df["longitude"], df["latitude"],
            c=df[cfg.target_col], cmap="plasma",
            s=4, alpha=0.6, linewidths=0,
            vmin=df[cfg.target_col].quantile(0.02),
            vmax=df[cfg.target_col].quantile(0.98),
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label(f"{cfg.target_col.capitalize()} (m)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

        ax.set_title("Spatial distribution of significant wave height",
                     fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        _save("spatial_distribution", cfg.output_dir)

    # Folium heatmap
    sample = df.sample(
        min(cfg.heatmap_samples, len(df)), random_state=cfg.random_state
    )
    m = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()],
        zoom_start=5,
    )
    cols = ["latitude", "longitude"] + (
        [cfg.target_col] if cfg.target_col in df.columns else []
    )
    HeatMap(sample[cols].values.tolist()).add_to(m)
    map_path = "map.html"
    m.save(map_path)
    print(f"  ✓ {map_path}")


def save_clean(df: pd.DataFrame, path: str = "cleaned_data.csv") -> None:
    df.to_csv(path, index=False)
    print(f"\n  ✓ Saved cleaned data → {path}")


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run(cfg: Optional[Config] = None) -> pd.DataFrame:
    cfg = cfg or Config()
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    _apply_theme()

    df = load(cfg)
    report_missing(df, cfg)
    df = clean(df, cfg)
    analyse_features(df, cfg)
    analyse_distributions(df, cfg)
    analyse_spatial(df, cfg)
    save_clean(df)

    print("\n══ EDA COMPLETE ══════════════════════════\n")
    return df


if __name__ == "__main__":
    run()