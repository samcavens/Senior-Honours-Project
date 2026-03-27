"""
Run sky-clarity analysis over many FITS images and save the results to a CSV.

For each FITS file in the chosen time window, this script:
1. Runs the SkyClarity analysis.
2. Extracts average Cvis, zero point, and per-region Cvis values.
3. Appends the results to a CSV file.
4. Skips files that are already present in the CSV.

This is a simple batch script used to process the full dataset.
"""

from pathlib import Path
import csv
import numpy as np

from Main import SkyClarity


# Input and output paths
raw_dir = Path("/path/to/raw_fits_directory")
output_csv = Path("/path/to/output_metrics.csv")

catalog_path = "/path/to/catalog_file.fits"
json_path = "/path/to/camera_calibration.json"
output_path = "/path/to/output_directory"

output_csv.parent.mkdir(parents=True, exist_ok=True)


def keep_night_file(fits_path):
    """
    Keep only files in the night-time window.

    Expected filename format:
    image-YYYYMMDD-HHMMSS.fits

    This keeps files from:
    18:30:00 to 23:59:59
    and
    00:00:00 to 04:00:00
    """
    time_str = fits_path.stem.split("-")[-1]
    file_time = int(time_str)

    return (file_time >= 183000) or (file_time <= 40000)


# HEALPix setup used by the sky-clarity code
nside = 2
npix = 12 * nside**2

header = (
    ["file", "avg_cvis", "avg_cvis_err", "zp", "zp_err", "zp_snr"]
    + [f"cvis_{i}" for i in range(npix)]
)

# Find all FITS files and keep only those in the desired time range
all_files = sorted(raw_dir.glob("*.fits"))
all_night_files = [f for f in all_files if keep_night_file(f)]

if not all_night_files:
    raise FileNotFoundError(f"No FITS files found in {raw_dir} for the chosen night-time window.")

# Read files already processed in the existing CSV
processed_files = set()

if output_csv.exists():
    with open(output_csv, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

        if rows:
            start_idx = 1 if rows[0] and rows[0][0] == "file" else 0
            for row in rows[start_idx:]:
                if row and row[0].strip():
                    processed_files.add(row[0].strip())

print(f"Already processed: {len(processed_files)} files")

# Only analyse files not already written to the CSV
file_list = [f for f in all_night_files if str(f) not in processed_files]
print(f"Remaining to process: {len(file_list)} files")

if not file_list:
    print("Nothing left to do.")
    raise SystemExit

# Append to the CSV if it exists, otherwise create it
write_header = not output_csv.exists() or output_csv.stat().st_size == 0
mode = "a" if output_csv.exists() and output_csv.stat().st_size > 0 else "w"

with open(output_csv, mode, newline="") as f:
    writer = csv.writer(f)

    if write_header:
        writer.writerow(header)
        f.flush()

    for i, fits_path in enumerate(file_list, 1):
        try:
            s = SkyClarity(
                str(fits_path),
                output_path,
                catalog_path,
                json_path,
            )

            avg_cvis, avg_cvis_err, cvis_arr, zp, zp_err, zp_snr = s.stars_and_cat(
                plot=False,
                plot2=False,
                plot3=False,
            )

            row = [
                str(fits_path),
                avg_cvis,
                avg_cvis_err,
                zp,
                zp_err,
                zp_snr,
            ] + cvis_arr.tolist()

            writer.writerow(row)
            f.flush()

            print(f"[{i}/{len(file_list)}] done: {fits_path.name}")

        except Exception as e:
            row = [
                str(fits_path),
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ] + [np.nan] * npix

            writer.writerow(row)
            f.flush()

            print(f"[{i}/{len(file_list)}] failed: {fits_path.name} -> {e}")

print(f"Saved results to {output_csv}")
