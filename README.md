# All-Sky Camera Sky Clarity Analysis

This project quantifies night sky clarity from all-sky camera images using star detection and catalogue matching. It was developed as part of a Senior Honours Physics project investigating the suitability of an astronomical site in Ileret, Kenya.

This repository contains the full analysis pipeline used to generate the results presented in the dissertation.

The pipeline combines astrometric calibration, source detection, and statistical analysis to estimate observational conditions.

---

## Overview

The workflow consists of three main stages:

1. **Astrometric Solving**
   - FITS images are cropped and solved using `astrometry.net`
   - Produces WCS-calibrated images

2. **Camera Calibration**
   - A physical model maps sky coordinates (altitude, azimuth) to image pixels
   - Model parameters are fitted using matched stars

3. **Sky Clarity Estimation**
   - Stars are detected in each image
   - Matched against a catalogue
   - Sky clarity metric \( C_{\mathrm{vis}} \) is computed across the sky

---

## Repository Structure

- `astronometry_solver.py`  
  Crops and solves FITS images using `astrometry.net`

- `Calibration.py`  
  Contains the camera model and calibration routines

- `Main.py`  
  Defines the `SkyClarity` class used to compute sky clarity metrics

- `run_batch.py`  
  Runs the full analysis over a directory of FITS images and saves results to CSV

- `README.md`  
  Project description and usage

---

## Requirements

Key Python packages:

- numpy  
- astropy  
- photutils  
- sep  
- matplotlib  
- scipy  
- astropy-healpix  

You also need:

- astrometry.net installed and available as `solve-field`

---

## Usage

### 1. Solve images

```bash
python astronometry_solver.py
```

---

### 2. Calibrate camera

```python
from Calibration import Calibrate, ImageParams

cal = Calibrate(fits_file, solved_file, output_dir, catalog_path)
cal.open()

init = ImageParams(
    xc=..., yc=...,
    s=..., theta_north_deg=...,
    k=[...], dr=[...], dt=[...]
)

best_model, diagnostics = cal.calibrate_two_stage(
    init_model=init,
    cfg_wcs=...,
    cfg_full=...,
    plot=True
)
```

This produces a JSON file containing the fitted camera model.

---

### 3. Compute sky clarity (single image)

```python
from Main import SkyClarity

s = SkyClarity(
    fits_file,
    output_dir,
    catalog_path,
    json_path
)

avg_cvis, avg_cvis_err, cvis_map, zp, zp_err, snr = s.stars_and_cat()
```

---

### 4. Run full dataset

```bash
python run_batch.py
```

This will:
- iterate through all FITS files in a directory  
- skip already processed files  
- compute sky clarity metrics  
- append results to a CSV file  

---

## Output

For each image the pipeline returns:

- Average sky clarity \( C_{\mathrm{vis}} \)  
- Per-region clarity values (HEALPix grid)  
- Photometric zero point (ZP)  
- Zero point uncertainty  
- Median signal-to-noise ratio  

---

## Method Summary

- Stars are detected using SEP with background subtraction  
- Catalogue stars are projected onto the image using a calibrated camera model  
- Matching is performed using nearest-neighbour methods  

\[
C_{\mathrm{vis}} = \frac{\text{weighted detected stars}}{\text{weighted catalogue stars}}
\]

- The sky is divided into equal-area regions using HEALPix  

---

## Notes

- FITS headers must contain:
  - DATE-OBS  
  - LATITUDE  
  - LONGITUDE  
  - ALTITUDE  

- A bright star catalogue (e.g. Johnson UBV) is required  
- Low-altitude regions (typically below ~25°) are excluded  
- Very bright or saturated frames are automatically rejected  

---

## Author

Sam Cavens  
Senior Honours Physics Project
