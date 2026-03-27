# Senior-Honours-Project

# All-Sky Camera Sky Clarity Analysis

This project quantifies night sky clarity from all-sky camera images using star detection and catalogue matching. It was developed as part of a Senior Honours Physics project investigating the suitability of an astronomical site in Ileret, Kenya.

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

- `numpy`
- `astropy`
- `photutils`
- `sep`
- `matplotlib`
- `scipy`
- `astropy-healpix`

You also need:

- `astrometry.net` installed and available as `solve-field`

---

## Usage

### 1. Solve images

Run the astrometry solver on your FITS files:

```bash
python astronometry_solver.py
