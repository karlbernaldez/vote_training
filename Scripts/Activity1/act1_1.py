from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

@dataclass
class Config:
    file_path: str = "../Dataset/flood/flood.csv"
    output_dir: str = "plots/flood_eda"
    max_pairplot_samples: int = 5_000
    outlier_trim: float = 0.01          # drop bottom/top 1%
    random_state: int = 42
    target_col: str = "FloodProbability"
    feature_cols: list[str] = field(default_factory=lambda: [
        "MonsoonIntensity",
        "TopographyDrainage",
        "RiverManagement",
        "Deforestation",
        "Urbanization",
        "ClimateChange",
        "DamsQuality",
        "Siltation",
        "AgriculturalPractices",
        "Encroachments",
        "IneffectiveDisasterPreparedness",
        "DrainageSystems",
        "CoastalVulnerability",
        "Landslides",
        "Watersheds",
        "DeterioratingInfrastructure",
        "PopulationScore",
        "WetlandLoss",
        "InadequatePlanning",
        "PoliticalFactors",
    ])
    critical_cols: list[str] = field(default_factory=lambda: [
        "FloodProbability",
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


def _save(name: str, output_dir: str) -> None:
    """Save the current figure and close it."""
    path = Path(output_dir) / f"{name}.png"
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
    return df


def report_missing(df: pd.DataFrame, cfg: Config) -> None:
    print("\n── 2. MISSING DATA ──────────────────────")
    missing_pct = df.isnull().mean().mul(100).sort_values(ascending=False)
    print(missing_pct.to_string())

    plt.figure(figsize=(14, 5))
    missing_pct.plot(kind="bar")
    plt.title("Missing Data %")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save("missing_data", cfg.output_dir)


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    print("\n── 3. CLEAN ─────────────────────────────")
    print(f"  Initial shape: {df.shape}")

    # 1. Drop fully-empty columns
    full_missing = df.columns[df.isnull().all()].tolist()
    if full_missing:
        print(f"  Dropping fully-missing cols: {full_missing}")
        df = df.drop(columns=full_missing)

    # 2. Duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df):,} duplicate rows")

    # 3. Drop rows missing critical columns
    critical = [c for c in cfg.critical_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"  Dropped {before - len(df):,} rows with missing critical values")

    # 4. Trim extreme outliers in target column
    if cfg.target_col in df.columns:
        lo = df[cfg.target_col].quantile(cfg.outlier_trim)
        hi = df[cfg.target_col].quantile(1 - cfg.outlier_trim)
        before = len(df)
        df = df[df[cfg.target_col].between(lo, hi)]
        print(f"  Removed {before - len(df):,} extreme outliers ({cfg.outlier_trim*100:.0f}% tails)")

    print(f"  Final shape:   {df.shape}")
    return df


def _drop_low_variance(df: pd.DataFrame, num_cols: pd.Index, threshold: float = 1e-6) -> pd.Index:
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

    plot_cols = _drop_low_variance(df, num_cols)
    if len(plot_cols) < 2:
        print("  Not enough variance in numeric columns to plot correlations.")
        return

    corr = df[plot_cols].corr()

    # — Full correlation heatmap —
    n = len(plot_cols)
    fig_size = max(10, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=1.2, linecolor="white",
        square=True, ax=ax,
        annot_kws={"size": 9, "weight": "bold"},
        cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    )
    ax.set_title("Pearson Correlation — Flood Features", fontsize=15, pad=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    _save("pearson_heatmap", cfg.output_dir)

    # — Feature correlation with target (bar chart) —
    if cfg.target_col in plot_cols:
        target_corr = (
            corr[cfg.target_col]
            .drop(cfg.target_col, errors="ignore")
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ["#e05c2a" if v >= 0 else "#4C72B0" for v in target_corr]
        target_corr.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Feature Correlation with {cfg.target_col}", fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel("Pearson r", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        sns.despine(fig=fig)
        fig.tight_layout()
        _save("target_correlation_bar", cfg.output_dir)

    # — Pairplot (top correlated features) —
    _pairplot(df, corr, plot_cols, cfg)


def _pairplot(
    df: pd.DataFrame,
    corr: pd.DataFrame,
    plot_cols: pd.Index,
    cfg: Config,
) -> None:
    target = cfg.target_col

    if target in df.columns and target in plot_cols:
        top = (
            corr[target].drop(target, errors="ignore")
            .abs().nlargest(4).index.tolist()
        )
        cols = [target] + [c for c in top if c in plot_cols]
    else:
        cols = list(plot_cols[:5])

    if len(cols) < 2:
        print("  Skipping pairplot — fewer than 2 plottable columns.")
        return

    sample = df[cols].dropna()
    if len(sample) > cfg.max_pairplot_samples:
        sample = sample.sample(cfg.max_pairplot_samples, random_state=cfg.random_state)

    colors = None
    norm = None
    if target in sample.columns:
        norm = plt.Normalize(sample[target].min(), sample[target].max())
        colors = plt.cm.plasma(norm(sample[target].values))

    n_cols = len(cols)
    g = sns.PairGrid(sample, corner=True, height=2.8, aspect=1.0, despine=True)

    def _scatter(x, y, **kwargs):
        kwargs.pop("color", None)
        plt.gca().scatter(
            x, y, c=colors, cmap="plasma",
            s=8, alpha=0.55, edgecolor="none", linewidths=0,
        )

    g.map_lower(_scatter)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.55, linewidth=1.2, warn_singular=False)

    for ax_row in g.axes:
        for ax in ax_row:
            if ax is not None:
                ax.tick_params(labelsize=9)
                ax.set_xlabel(ax.get_xlabel(), fontsize=9)
                ax.set_ylabel(ax.get_ylabel(), fontsize=9)

    if norm is not None:
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        last_diag = g.axes[n_cols - 1][n_cols - 1]
        if last_diag is not None:
            cbar = g.figure.colorbar(sm, ax=last_diag, fraction=0.08, pad=0.04, shrink=0.8)
            cbar.set_label(target, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

    g.figure.suptitle("Pairplot — Top Features vs FloodProbability", y=1.01, fontsize=13, fontweight="bold")
    g.figure.tight_layout()
    path = Path(cfg.output_dir) / "pairplot_top_features.png"
    g.figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")


def analyse_distributions(df: pd.DataFrame, cfg: Config) -> None:
    print("\n── 5. DISTRIBUTIONS ─────────────────────")
    target = cfg.target_col

    # — FloodProbability distribution —
    if target in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(
            df[target], bins=50, kde=True, ax=ax,
            color="#4C72B0", alpha=0.55,
            line_kws={"linewidth": 2.2, "color": "#1a3a6b"},
        )
        median_val = df[target].median()
        ax.axvline(median_val, color="#e05c2a", linewidth=1.8, linestyle="--",
                   label=f"Median {median_val:.2f}")
        ax.set_title(f"Distribution of {target}", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel(target, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        sns.despine(fig=fig)
        fig.tight_layout()
        _save("flood_probability_distribution", cfg.output_dir)

    # — Feature distributions (all features, grid layout) —
    feature_cols = [c for c in cfg.feature_cols if c in df.columns]
    if feature_cols:
        n = len(feature_cols)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
        axes = axes.flatten()

        for i, col in enumerate(feature_cols):
            sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i],
                         color="#4C72B0", alpha=0.5,
                         line_kws={"linewidth": 1.5, "color": "#1a3a6b"})
            axes[i].set_title(col, fontsize=10, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].tick_params(labelsize=8)
            sns.despine(ax=axes[i])

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        _save("feature_distributions_grid", cfg.output_dir)

    # — Box plots: feature vs FloodProbability risk buckets —
    if target in df.columns and feature_cols:
        df = df.copy()
        df["RiskBucket"] = pd.cut(
            df[target],
            bins=[0, 0.3, 0.6, 1.01],
            labels=["Low", "Medium", "High"],
            right=False,
        )
        ncols = 4
        nrows = int(np.ceil(len(feature_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
        axes = axes.flatten()
        palette = {"Low": "#4C72B0", "Medium": "#f0a500", "High": "#e05c2a"}

        for i, col in enumerate(feature_cols):
            sns.boxplot(
                data=df, x="RiskBucket", y=col,
                order=["Low", "Medium", "High"],
                palette=palette, ax=axes[i],
                linewidth=1.2, fliersize=2,
            )
            axes[i].set_title(col, fontsize=10, fontweight="bold")
            axes[i].set_xlabel("")
            axes[i].tick_params(labelsize=8)
            sns.despine(ax=axes[i])

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Feature Distribution by Flood Risk Bucket", fontsize=14,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        _save("feature_boxplots_by_risk", cfg.output_dir)


def analyse_risk_factors(df: pd.DataFrame, cfg: Config) -> None:
    """Flood-specific analysis: mean feature values by risk level."""
    print("\n── 6. RISK FACTOR ANALYSIS ──────────────")
    target = cfg.target_col
    feature_cols = [c for c in cfg.feature_cols if c in df.columns]

    if target not in df.columns or not feature_cols:
        print("  Skipped — target or feature columns not found.")
        return

    df = df.copy()
    df["RiskBucket"] = pd.cut(
        df[target],
        bins=[0, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"],
        right=False,
    )

    # Mean feature values per risk bucket (normalised 0-1 for comparability)
    means = df.groupby("RiskBucket", observed=True)[feature_cols].mean()
    means_norm = (means - means.min()) / (means.max() - means.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 6))
    means_norm.T.plot(kind="bar", ax=ax, colormap="RdYlBu_r", edgecolor="white", width=0.75)
    ax.set_title("Normalised Mean Feature Values by Flood Risk Bucket",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Normalised Mean (0–1)", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.legend(title="Risk", fontsize=10)
    sns.despine(fig=fig)
    fig.tight_layout()
    _save("risk_factor_means", cfg.output_dir)

    # Top 5 drivers (highest mean in "High" risk bucket)
    if "High" in means.index:
        top5 = means.loc["High"].nlargest(5)
        fig, ax = plt.subplots(figsize=(8, 5))
        top5.sort_values().plot(kind="barh", ax=ax, color="#e05c2a", edgecolor="white")
        ax.set_title("Top 5 Drivers in High-Risk Areas", fontsize=14, fontweight="bold", pad=10)
        ax.set_xlabel("Mean Feature Value", fontsize=11)
        sns.despine(fig=fig)
        fig.tight_layout()
        _save("top5_high_risk_drivers", cfg.output_dir)


def save_clean(df: pd.DataFrame, path: str = "cleaned_flood_data.csv") -> None:
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
    analyse_risk_factors(df, cfg)
    save_clean(df)

    print("\n══ EDA COMPLETE ══════════════════════════\n")
    return df


if __name__ == "__main__":
    run()