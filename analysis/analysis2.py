from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# PATHS
csv_path = Path("/Users/samcavens/Documents/4th/SHP/test_metrics4.csv")
output_dir = Path("/Users/samcavens/Documents/4th/SHP/monthly_metric_plots")
output_dir.mkdir(parents=True, exist_ok=True)

# LOAD + BASIC PROCESSING
df = pd.read_csv(csv_path)

df["timestamp_str"] = df["file"].astype(str).str.extract(r"image-(\d{8}-\d{6})\.fits")
df["timestamp"] = pd.to_datetime(
    df["timestamp_str"],
    format="%Y%m%d-%H%M%S",
    errors="coerce"
)

df = df.dropna(subset=["timestamp"]).copy()

df["month_num"] = df["timestamp"].dt.month
df["month_name"] = df["timestamp"].dt.strftime("%b")
df["hour"] = df["timestamp"].dt.hour
df["date"] = df["timestamp"].dt.date

df["avg_cvis"] = pd.to_numeric(df["avg_cvis"], errors="coerce")
df["zp"] = pd.to_numeric(df["zp"], errors="coerce")
df["zp_snr"] = pd.to_numeric(df["zp_snr"], errors="coerce")

df["cvis_valid"] = np.isfinite(df["avg_cvis"])
df["zp_valid"] = np.isfinite(df["zp"])
df["snr_valid"] = np.isfinite(df["zp_snr"])

# SEASONS
def assign_season(month):
    if month in [1, 2]:
        return "Hot dry\n(Jan-Feb)"
    elif month in [3, 4, 5]:
        return "Long rains\n(Mar-May)"
    elif month in [6, 7, 8, 9]:
        return "Cool dry\n(Jun-Sep)"
    else:
        return "Short rains\n(Oct-Dec)"

df["season"] = df["month_num"].apply(assign_season)

season_order = [
    "Hot dry\n(Jan-Feb)",
    "Long rains\n(Mar-May)",
    "Cool dry\n(Jun-Sep)",
    "Short rains\n(Oct-Dec)"
]

month_order = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# MONTHLY METRIC STATS
def monthly_stats(df_in, column):
    m = df_in.groupby(["month_num","month_name"])[column].agg(
        mean="mean", median="median", std="std", count="count"
    ).reset_index()
    m = m.sort_values("month_num")
    m["sem"] = m["std"] / np.sqrt(m["count"])
    return m

monthly_cvis = monthly_stats(df[df["cvis_valid"]], "avg_cvis")
monthly_zp   = monthly_stats(df[df["zp_valid"]], "zp")
monthly_snr  = monthly_stats(df[df["snr_valid"]], "zp_snr")

monthly_total = df.groupby(["month_num","month_name"]).agg(
    total_images=("file","count"),
    cvis_valid_count=("cvis_valid","sum"),
    zp_valid_count=("zp_valid","sum"),
    snr_valid_count=("snr_valid","sum")
).reset_index().sort_values("month_num")

monthly_total["cvis_frac"] = monthly_total["cvis_valid_count"] / monthly_total["total_images"]
monthly_total["zp_frac"]   = monthly_total["zp_valid_count"] / monthly_total["total_images"]
monthly_total["snr_frac"]  = monthly_total["snr_valid_count"] / monthly_total["total_images"]

monthly_zp_cov = monthly_total.merge(
    monthly_zp[["month_num","month_name","mean","sem"]],
    on=["month_num","month_name"],
    how="left"
).rename(columns={"mean":"zp_mean","sem":"zp_sem"})

monthly_snr_cov = monthly_total.merge(
    monthly_snr[["month_num","month_name","mean","sem"]],
    on=["month_num","month_name"],
    how="left"
).rename(columns={"mean":"snr_mean","sem":"snr_sem"})

def plot_metric_with_coverage(monthly_cov, value_col, err_col, frac_col, ylabel_left, save_name):
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax2 = ax1.twinx()
    ax2.bar(monthly_cov["month_name"], 100*monthly_cov[frac_col], alpha=0.3)
    ax2.set_ylabel("Measurement coverage (%)")
    ax2.set_ylim(0,100)

    ax1.errorbar(
        monthly_cov["month_name"],
        monthly_cov[value_col],
        yerr=monthly_cov[err_col],
        fmt="o-",
        capsize=4
    )

    ax1.set_xlabel("Month")
    ax1.set_ylabel(ylabel_left)

    plt.tight_layout()
    plt.savefig(output_dir / save_name, dpi=300)
    plt.show()

plot_metric_with_coverage(monthly_zp_cov, "zp_mean", "zp_sem", "zp_frac",
                          "Mean Zero Point (mag)", "monthly_zp.png")

plot_metric_with_coverage(monthly_snr_cov, "snr_mean", "snr_sem", "snr_frac",
                          "Mean ZP SNR", "monthly_snr.png")

# SEASONAL HOURLY CVIS
night_hours = [18,19,20,21,22,23,0,1,2]
df_cvis_night = df[(df["cvis_valid"]) & (df["hour"].isin(night_hours))].copy()

hourly = df_cvis_night.groupby(["season","hour"])["avg_cvis"].agg(
    mean="mean", std="std", count="count"
).reset_index()

hourly["sem"] = hourly["std"] / np.sqrt(hourly["count"])

hour_map = {18:18,19:19,20:20,21:21,22:22,23:23,0:24,1:25,2:26}
hourly["plot_hour"] = hourly["hour"].map(hour_map)

def plot_hourly(hourly):
    fig, axes = plt.subplots(2,2, figsize=(14,9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, season in zip(axes, season_order):
        sub = hourly[hourly["season"] == season].dropna()

        ax.errorbar(sub["plot_hour"], sub["mean"], yerr=sub["sem"], fmt="o-", capsize=4)
        ax.set_title(season)
        ax.set_xlim(17.5,26.5)
        ax.set_xticks([18,19,20,21,22,23,24,25,26])
        ax.set_xticklabels(["18","19","20","21","22","23","0","1","2"])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "seasonal_hourly_cvis.png", dpi=300)
    plt.show()

plot_hourly(hourly)

# SEASONAL MEAN CVIS
df_cvis = df[df["cvis_valid"]]

seasonal = df_cvis.groupby("season")["avg_cvis"].agg(
    mean="mean", std="std", count="count"
).reindex(season_order).reset_index()

seasonal["sem"] = seasonal["std"] / np.sqrt(seasonal["count"])

plt.figure(figsize=(8,5))
plt.errorbar(seasonal["season"], seasonal["mean"], yerr=seasonal["sem"], fmt="o-", capsize=5)
plt.ylabel("Mean Cvis")
plt.xlabel("Season")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "seasonal_mean_cvis.png", dpi=300)
plt.show()

# MONTE CARLO NIGHT CLASSIFICATION
n_iter = 1000
rng = np.random.default_rng(42)

df_mc_base = df_cvis.copy()
df_mc_base = df_mc_base[(df_mc_base["avg_cvis"] >= 0) & (df_mc_base["avg_cvis"] <= 1)]

df_mc_base["avg_cvis_err"] = 0.05

results = []

for i in range(n_iter):
    df_mc = df_mc_base.copy()

    df_mc["cvis_mc"] = rng.normal(df_mc["avg_cvis"], df_mc["avg_cvis_err"])
    df_mc["cvis_mc"] = df_mc["cvis_mc"].clip(0,1)

    nightly = df_mc.groupby(["date","month_name"]).agg(
        n04=("cvis_mc", lambda x: np.sum(x > 0.4)),
        n06=("cvis_mc", lambda x: np.sum(x > 0.6))
    ).reset_index()

    nightly["spec"] = nightly["n04"] >= 25
    nightly["phot"] = nightly["n06"] >= 60

    monthly = nightly.groupby("month_name").agg(
        spec_pct=("spec","mean"),
        phot_pct=("phot","mean")
    ).reset_index()

    results.append(monthly)

mc = pd.concat(results)

summary = mc.groupby("month_name").agg(
    phot_mean=("phot_pct","mean"),
    phot_std=("phot_pct","std"),
    spec_mean=("spec_pct","mean"),
    spec_std=("spec_pct","std")
).reindex(month_order).reset_index()

print(summary)

plt.figure(figsize=(9,5))
x = np.arange(len(summary))

plt.bar(x, summary["phot_mean"], label="photometric")
plt.bar(x, summary["spec_mean"] - summary["phot_mean"],
        bottom=summary["phot_mean"], label="spectroscopic")

plt.xticks(x, summary["month_name"])
plt.ylabel("Fraction of usable nights")
plt.legend()
plt.tight_layout()

plt.savefig(output_dir / "monthly_usable_nights.png", dpi=300)
plt.show()
