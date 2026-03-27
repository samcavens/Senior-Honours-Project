#!/usr/bin/env python3
"""
Batch-solve all-sky FITS images with astrometry.net.

For each FITS file in raw_dir, the script:
1. Opens the image and header.
2. Makes a circular central crop to focus the solve on the useful sky region.
3. Estimates the RA/Dec of the zenith from the FITS header time and location.
4. Runs astrometry.net solve-field using that RA/Dec as a starting guess.
5. Renames the solved output and removes intermediate files.

Just set raw_dir and solved_dir below, then run the file.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord


class Solve:
    """Open, crop, and astrometrically solve one FITS image."""

    def __init__(self, file_path, solved_dir):
        # Store the input file path and output folder.
        self.file_path = Path(file_path)
        self.solved_dir = Path(solved_dir)
        self.solved_dir.mkdir(exist_ok=True)

    def open(self):
        """Read the image data and FITS header from disk."""
        with fits.open(self.file_path) as hdul:
            self.data = hdul[0].data
            self.header = hdul[0].header.copy()

    def circular_crop(self, crop_radius_frac=0.17, dx=0.0, dy=0.0):
        """
        Make a circular crop around the image centre.

        Parameters:

        crop_radius_frac : float
            Radius of the crop as a fraction of the smaller image dimension.
        dx, dy : float
            Optional shifts of the crop centre in pixels.

        Notes:
        Pixels outside the circular region are filled with the median value
        of the crop. This keeps the image rectangular for astrometry.net
        while masking unwanted outer regions.
        """
        ny, nx = self.data.shape

        # Start from the image centre, then apply any manual offset.
        cx = (nx - 1) / 2 + dx
        cy = (ny - 1) / 2 + dy
        r = crop_radius_frac * min(ny, nx)

        # Bounding square around the circle.
        x0 = max(0, int(cx - r))
        x1 = min(nx, int(cx + r))
        y0 = max(0, int(cy - r))
        y1 = min(ny, int(cy + r))

        # Extract the square cutout.
        cut = self.data[y0:y1, x0:x1].copy()
        bkg = np.nanmedian(cut)

        # Build a circular mask inside the cutout.
        yy, xx = np.indices(cut.shape)
        mask = (xx - (cx - x0))**2 + (yy - (cy - y0))**2 <= r**2

        # Replace pixels outside the circle with the median background.
        cut[~mask] = bkg

        # Save crop information in the Fits header for later reference.
        self.header["HIERARCH CUT_X0"] = x0
        self.header["HIERARCH CUT_Y0"] = y0
        self.header["HIERARCH CIRC_CX"] = float(cx)
        self.header["HIERARCH CIRC_CY"] = float(cy)
        self.header["HIERARCH CIRC_R"] = float(r)

        # Write the cropped image to disk.
        self.crop_path = self.solved_dir / f"{self.file_path.stem}_crop.fits"
        fits.writeto(self.crop_path, cut, self.header, overwrite=True)

    def wcs_guess(self):
        """
        Estimate the RA and Dec of the zenith.

        The FITS header gives the observation time and site location.
        From this, the sky coordinates at altitude = 90 deg are found
        and used as a starting guess for astrometry.net.
        """
        t = Time(self.header["DATE-OBS"], scale="utc")
        lon = self.header.get("LONGITUD", self.header.get("LONGITUDE"))
        lat = self.header["LATITUDE"]
        height = self.header.get("ALTITUDE", 0.0)

        loc = EarthLocation(
            lat=float(lat) * u.deg,
            lon=float(lon) * u.deg,
            height=float(height) * u.m,
        )

        zenith = SkyCoord(
            alt=90 * u.deg,
            az=0 * u.deg,
            frame=AltAz(obstime=t, location=loc),
        ).icrs

        return zenith.ra.deg, zenith.dec.deg

    def solve(self, radius_deg=60, timeout=120, cpulimit=60):
        """
        Run astrometry.net solve-field on the cropped image.

        Parameters
        radius_deg : float
            Search radius in degrees around the RA/Dec guess.
        timeout : int
            Maximum wall time allowed for solve-field.
        cpulimit : int
            CPU time limit passed to solve-field.
        """
        ra, dec = self.wcs_guess()

        # Command passed to astrometry.net.
        cmd = [
            shutil.which("solve-field") or "solve-field",
            str(self.crop_path),
            "--overwrite",
            "--no-plots",
            "--guess-scale",
            "--crpix-center",
            "--cpulimit", str(cpulimit),
            "--ra", str(ra),
            "--dec", str(dec),
            "--radius", str(radius_deg),
        ]

        subprocess.run(cmd, check=True, timeout=timeout)

        # astrometry.net writes the solved FITS as a .new file.
        new_path = self.crop_path.with_suffix(".new")
        solved_path = self.crop_path.with_name(f"{self.file_path.stem}_crop_solved.fits")

        if new_path.exists():
            new_path.rename(solved_path)

        # Remove intermediate astrometry.net files.
        for ext in [".axy", ".corr", ".match", ".rdls", ".solved", ".xyls", ".wcs"]:
            p = self.crop_path.with_suffix(ext)
            if p.exists():
                p.unlink()

        # Remove the temporary cropped FITS file.
        if self.crop_path.exists():
            self.crop_path.unlink()


# Set input and output folders here.
raw_dir = Path("/path/to/raw_dir")
solved_dir = Path("/path/to/solved_dir")

# Find all FITS files in the raw directory.
fits_files = sorted(raw_dir.glob("*.fits"))

for i, fits_path in enumerate(fits_files, 1):
    out_path = solved_dir / f"{fits_path.stem}_crop_solved.fits"

    # Skip files that have already been solved.
    if out_path.exists():
        print(f"[{i}/{len(fits_files)}] skip {fits_path.name}")
        continue

    print(f"[{i}/{len(fits_files)}] solving {fits_path.name}")

    try:
        img = Solve(fits_path, solved_dir)
        img.open()
        img.circular_crop(crop_radius_frac=0.17, dx=0.0, dy=0.0)
        img.solve(radius_deg=60, timeout=120, cpulimit=60)
        print("  success")
    except Exception as e:
        print(f"  failed: {e}")
