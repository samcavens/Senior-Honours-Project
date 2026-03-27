"""
Make summary plots from the batch sky-clarity CSV output.

This script:
1. Loads the CSV produced by run_batch.py
2. Extracts timestamps from FITS filenames
3. Computes monthly statistics for Cvis, zero point, and SNR
4. Makes monthly trend and distribution plots
5. Computes moon illumination and compares it with daily average Cvis

This is an analysis script for plotting results, not part of the core pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from astropy.time import Time
from astropy.coordinates import get_body
import astropy.units as u


# Paths
csv_path = Path("/path/to/metrics.csv")
output_dir = Path("/path/to/analysis_outputs")
output_dir.mkdir(parents=True, exist_ok=True)


def extract_datetime(file_path):
    """
    Extract the observation timestamp from a filename of the form:
    image-YYYYMMDD-HHMMSS.fits
    """
    try:
        name = Path(file_path).name
        date_str = name.replace("image-", "").replace(".fits", "")
        return pd.to_datetime(date_str, format="%Y%m%d-%H%M%S")
    except Exception:
        return pd.NaT


def monthly_stats(df_in, column):
    """
    Compute monthly mean, median, standard deviation, count, and SEM
    for one metric.
    """
    monthly = df_in.groupby(["month_num", "month_name"])[column].agg(
        mean="mean",
        median="median",
        std="std",
        count="count"
    ).reset_index()

    monthly = monthly.sort_values("month_num")
    monthly["sem"] = monthly["std"] / np.sqrt(monthly["count"])
    return monthly


def moon_illumination(times):
    """
    Compute moon illumination fraction for a sequence of datetimes.
    """
    t = Time(times.to_numpy())
    sun = get_body("sun", t)
    moon = get_body("moon", t)

    phase_angle = sun.separation(moon)
    cos_phase = np.cos(phase_angle.to(u.rad).value)
    illum = (1 - cos_phase) / 2
    return illum


def plot_cvis_mean_and_median(monthly):
    """
    Plot monthly mean ± SEM and monthly median for Cvis.
    """
    plt.figure(figsize=(5, 5))

    plt.errorbar(
        monthly["month_name"],
        monthly["mean"],
        yerr=monthly["sem"],
        fmt="o-",
        capsize=4,
        label="Mean ± SEM"
    )

    plt.plot(
        monthly["month_name"],
        monthly["median"],
        marker="o",
        label="Median"
    )

    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Cvis (0–1)", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "cvis_mean_and_median.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_metric(monthly, ylabel, save_name):
    """
    Plot monthly mean ± SEM for a general metric.
    """
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        monthly["month_name"],
        monthly["mean"],
        yerr=monthly["sem"],
        fmt="o-",
        capsize=4,
        label="Mean ± SEM"
    )

    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / save_name, dpi=300, bbox_inches="tight")
    plt.show()


def plot_cvis_boxplot(df_cvis):
    """
    Plot monthly Cvis distributions after 3-sigma clipping within each month.
    """
    clipped_groups = []

    for month, group in df_cvis.groupby("month_num"):
        mu = group["avg_cvis"].mean()
        sigma = group["avg_cvis"].std()

        if pd.isna(sigma) or sigma == 0:
            clipped = group.copy()
        else:
            clipped = group[
                (group["avg_cvis"] >= mu - 3 * sigma) &
                (group["avg_cvis"] <= mu + 3 * sigma)
            ].copy()

        clipped_groups.append(clipped)

    df_cvis_3std = pd.concat(clipped_groups, ignore_index=True)

    plt.figure(figsize=(10, 5))
    df_cvis_3std.boxplot(column="avg_cvis", by="month_num", grid=False)

    median_line = mlines.Line2D([], [], color="green", label="Median")
    box_patch = mpatches.Patch(facecolor="lightblue", label="Interquartile range")
    whisker_line = mlines.Line2D([], [], color="black", linestyle="-", label="Range")

    plt.legend(handles=[median_line, box_patch, whisker_line], loc="upper right")
    plt.title("Cvis distribution by month (3σ-clipped)")
    plt.suptitle("")
    plt.xlabel("Month")
    plt.ylabel("Cvis")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_cvis_boxplot.png", dpi=300)
    plt.show()


def plot_zp_boxplot(df_zp):
    """
    Plot monthly zero-point distributions.
    """
    plt.figure(figsize=(10, 5))
    df_zp.boxplot(column="zp", by="month_num", grid=False)

    plt.title("Zero point distribution by month")
    plt.suptitle("")
    plt.xlabel("Month")
    plt.ylabel("Zero point")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_zp_boxplot.png", dpi=300)
    plt.show()


def plot_cvis_violin(df_cvis):
    """
    Plot monthly Cvis distributions as violins.
    """
    plt.figure(figsize=(5, 5))

    data = [
        df_cvis.loc[df_cvis["month_num"] == m, "avg_cvis"].dropna()
        for m in range(1, 13)
    ]

    parts = plt.violinplot(
        data,
        positions=range(1, 13),
        showmeans=False,
        showmedians=True
    )

    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)

    parts["cmedians"].set_color("orange")
    parts["cmedians"].set_linewidth(2)

    plt.xticks(
        range(1, 13),
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )

    plt.ylabel("Cvis (0–1)", fontsize=12)
    plt.xlabel("Month", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_cvis_violin.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_smoothed_monthly_trends(monthly_cvis, monthly_zp):
    """
    Plot 3-month smoothed trends for Cvis and zero point.
    """
    monthly_cvis_mean = monthly_cvis["mean"]
    smoothed_cvis = monthly_cvis_mean.rolling(3, center=True).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(monthly_cvis_mean.index, monthly_cvis_mean, label="Mean Cvis", alpha=0.5)
    plt.plot(smoothed_cvis.index, smoothed_cvis, linewidth=2, label="3-month smooth")
    plt.xlabel("Month")
    plt.ylabel("Cvis")
    plt.xticks(range(len(monthly_cvis["month_num"])), monthly_cvis["month_num"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_cvis_smoothed.png", dpi=300)
    plt.show()

    monthly_zp_mean = monthly_zp["mean"]
    smoothed_zp = monthly_zp_mean.rolling(3, center=True).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(monthly_zp_mean.index, monthly_zp_mean, label="Mean ZP", alpha=0.5)
    plt.plot(smoothed_zp.index, smoothed_zp, linewidth=2, label="3-month smooth")
    plt.xlabel("Month")
    plt.ylabel("Zero point")
    plt.xticks(range(len(monthly_zp["month_num"])), monthly_zp["month_num"])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_zp_smoothed.png", dpi=300)
    plt.show()


def plot_combined_monthly(monthly_cvis, monthly_zp):
    """
    Plot median monthly Cvis and median monthly zero point on twin axes.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(monthly_cvis["month_num"], monthly_cvis["median"], marker="o", label="Median Cvis")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Cvis")
    ax1.set_xticks(range(1, 13))
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(monthly_zp["month_num"], monthly_zp["median"], marker="s", linestyle="--", label="Median ZP")
    ax2.set_ylabel("Zero point")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.tight_layout()
    plt.savefig(output_dir / "monthly_cvis_zp_combined.png", dpi=300)
    plt.show()


def plot_daily_cvis_vs_moon(df, start_date="2023-10-01", end_date="2024-03-01"):
    """
    Plot daily average Cvis together with moon illumination.
    """
    df_plot = df.copy()
    df_plot = df_plot[(df_plot["datetime"] >= start_date) & (df_plot["datetime"] < end_date)].copy()

    print("Computing moon illumination...")
    df_plot["moon_illum"] = moon_illumination(df_plot["datetime"])

    df_plot["date"] = df_plot["datetime"].dt.floor("D")

    daily = (
        df_plot.groupby("date")[["avg_cvis", "moon_illum"]]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    daily["avg_cvis_7d"] = daily["avg_cvis"].rolling(window=7, center=True, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(10, 7), constrained_layout=True)

    ax1.plot(
        daily["date"],
        daily["avg_cvis_7d"],
        linewidth=2.4,
        label="7-day rolling mean Cvis"
    )

    ax1.set_xlabel("Date", fontsize=14)
    ax1.set_ylabel("Average Cvis", fontsize=14)
    ax1.set_ylim(0, 0.9)

    ax2 = ax1.twinx()
    ax2.plot(
        daily["date"],
        daily["moon_illum"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        color="orange",
        label="Moon illumination"
    )

    ax2.set_ylabel("Moon illumination (0–1)", fontsize=14)
    ax2.set_ylim(-0.02, 1.05)

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax1.get_xticklabels(), rotation=0, ha="center")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=13)

    plt.savefig(output_dir / "daily_cvis_moon.png", dpi=300, bbox_inches="tight")
    plt.show()


# Load data
df = pd.read_csv(csv_path, low_memory=False)

# Parse datetime from filename
df["datetime"] = df["file"].apply(extract_datetime)
df = df.dropna(subset=["datetime"]).copy()

# Numeric conversion
df["avg_cvis"] = pd.to_numeric(df["avg_cvis"], errors="coerce")
df["zp"] = pd.to_numeric(df["zp"], errors="coerce")
df["zp_snr"] = pd.to_numeric(df["zp_snr"], errors="coerce")

# Add month columns
df["month_num"] = df["datetime"].dt.month
df["month_name"] = df["datetime"].dt.month_name().str[:3]

# Split by valid metric
df_cvis = df[np.isfinite(df["avg_cvis"])].copy()
df_cvis = df_cvis[(df_cvis["avg_cvis"] >= 0) & (df_cvis["avg_cvis"] <= 1)]

df_zp = df[np.isfinite(df["zp"])].copy()
df_snr = df[np.isfinite(df["zp_snr"])].copy()

# Monthly summaries
monthly_cvis = monthly_stats(df_cvis, "avg_cvis")
monthly_zp = monthly_stats(df_zp, "zp")
monthly_snr = monthly_stats(df_snr, "zp_snr")

monthly_cvis.to_csv(output_dir / "monthly_cvis_stats.csv", index=False)
monthly_zp.to_csv(output_dir / "monthly_zp_stats.csv", index=False)

coverage = pd.DataFrame({
    "month_num": range(1, 13)
}).merge(
    monthly_cvis[["month_num", "count"]].rename(columns={"count": "cvis_count"}),
    on="month_num",
    how="left"
).merge(
    monthly_zp[["month_num", "count"]].rename(columns={"count": "zp_count"}),
    on="month_num",
    how="left"
).fillna(0)

coverage.to_csv(output_dir / "monthly_counts_comparison.csv", index=False)

# Make plots
plot_cvis_mean_and_median(monthly_cvis)
plot_metric(monthly_zp, "Zero point (mag)", "monthly_zp_mean_sem.png")
plot_metric(monthly_snr, "ZP signal-to-noise", "monthly_snr_mean_sem.png")
plot_cvis_boxplot(df_cvis)
plot_zp_boxplot(df_zp)
plot_cvis_violin(df_cvis)
plot_smoothed_monthly_trends(monthly_cvis, monthly_zp)
plot_combined_monthly(monthly_cvis, monthly_zp)
plot_daily_cvis_vs_moon(df)

# Print summaries
print("\nCvis:\n", monthly_cvis)
print("\nZero point:\n", monthly_zp)
print("\nZP SNR:\n", monthly_snr)

print("\nCounts:")
print("Total rows with valid timestamp:", len(df))
print("Rows with valid Cvis:", len(df_cvis))
print("Rows with valid ZP:", len(df_zp))
print("Rows with valid ZP SNR:", len(df_snr))

print(f"\nOutputs saved to: {output_dir.resolve()}")
