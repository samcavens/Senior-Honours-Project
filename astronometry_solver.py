#!/usr/bin/env python3
"""
Solve all-sky FITS images with astrometry.net after applying a central crop.

Examples
--------
Solve one file with a circular crop:
    python solve_fits.py single /path/to/image.fits /path/to/solved_dir --crop-radius-frac 0.17 --radius-deg 60

Batch solve a directory:
    python solve_fits.py batch /path/to/raw_dir /path/to/solved_dir --crop-radius-frac 0.17 --radius-deg 60
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time


logger = logging.getLogger(__name__)


class Solve:
    """Handle reading, cropping, and solving a FITS all-sky image."""

    def __init__(self, file_path: str | Path, solved_dir: str | Path) -> None:
        self.file_path = Path(file_path)
        self.solved_dir = Path(solved_dir)
        self.solved_dir.mkdir(parents=True, exist_ok=True)

        self.header: Optional[fits.Header] = None
        self.image_data: Optional[np.ndarray] = None

        # Populated by open()
        self.time: Optional[Time] = None
        self.location: Optional[EarthLocation] = None

        # Populated by crop methods
        self.cropped_data: Optional[np.ndarray] = None
        self.cropped_header: Optional[fits.Header] = None
        self.cropped_path: Optional[Path] = None

        # Populated by wcs_guess()
        self.ra_deg: Optional[float] = None
        self.dec_deg: Optional[float] = None

    def open(self) -> tuple[Time, EarthLocation]:
        """Open FITS image and read observation time/location from header."""
        with fits.open(self.file_path) as hdulist:
            primary_hdu = hdulist[0]
            self.header = primary_hdu.header.copy()
            self.image_data = np.asarray(primary_hdu.data)

        if self.image_data is None:
            raise ValueError(f"No image data found in {self.file_path}")

        if self.image_data.ndim != 2:
            raise ValueError(
                f"Expected a 2D FITS image, got shape {self.image_data.shape} in {self.file_path}"
            )

        date_obs = self.header.get("DATE-OBS")
        lon = self.header.get("LONGITUD", self.header.get("LONGITUDE"))
        lat = self.header.get("LATITUDE")
        height = self.header.get("ALTITUDE", 0.0)

        if date_obs is None:
            raise KeyError(f"Missing DATE-OBS in FITS header: {self.file_path}")
        if lon is None:
            raise KeyError(f"Missing LONGITUD/LONGITUDE in FITS header: {self.file_path}")
        if lat is None:
            raise KeyError(f"Missing LATITUDE in FITS header: {self.file_path}")

        try:
            t = Time(date_obs, format="isot", scale="utc")
        except ValueError:
            # fallback in case header format is slightly different
            t = Time(date_obs, scale="utc")

        loc = EarthLocation(
            lat=float(lat) * u.deg,
            lon=float(lon) * u.deg,
            height=float(height) * u.m,
        )

        self.time = t
        self.location = loc

        return t, loc

    def wcs_guess(self) -> tuple[float, float]:
        """Estimate RA/Dec at zenith for use as an astrometry.net guess."""
        if self.time is None or self.location is None:
            raise RuntimeError("Call open() first to populate time and location.")

        zenith_altaz = SkyCoord(
            alt=90 * u.deg,
            az=0 * u.deg,
            frame=AltAz(obstime=self.time, location=self.location),
        )
        zenith_icrs = zenith_altaz.icrs

        self.ra_deg = float(zenith_icrs.ra.deg)
        self.dec_deg = float(zenith_icrs.dec.deg)

        logger.info("Zenith RA (deg): %.6f", self.ra_deg)
        logger.info("Zenith Dec (deg): %.6f", self.dec_deg)
        logger.info(
            "Zenith RA (hms): %s",
            zenith_icrs.ra.to_string(unit=u.hour, sep=":"),
        )
        logger.info(
            "Zenith Dec (dms): %s",
            zenith_icrs.dec.to_string(unit=u.deg, sep=":"),
        )

        return self.ra_deg, self.dec_deg

    def square_crop(self, crop_frac: float, plot: bool = False) -> Path:
        """Write a central square crop of the image."""
        if self.image_data is None or self.header is None:
            raise RuntimeError("Call open() before cropping.")

        if not (0.0 <= crop_frac < 0.5):
            raise ValueError("crop_frac must satisfy 0 <= crop_frac < 0.5")

        ny, nx = self.image_data.shape

        y0, y1 = int(ny * crop_frac), int(ny * (1 - crop_frac))
        x0, x1 = int(nx * crop_frac), int(nx * (1 - crop_frac))

        cropped_image = self.image_data[y0:y1, x0:x1]
        cropped_header = self.header.copy()
        cropped_header["NAXIS1"] = cropped_image.shape[1]
        cropped_header["NAXIS2"] = cropped_image.shape[0]
        cropped_header["HIERARCH CUT_X0"] = (x0, "cutout origin x (orig pix)")
        cropped_header["HIERARCH CUT_Y0"] = (y0, "cutout origin y (orig pix)")

        crop_output_path = self.solved_dir / f"{self.file_path.stem}_crop.fits"

        fits.writeto(crop_output_path, cropped_image, cropped_header, overwrite=True)

        self.cropped_data = cropped_image
        self.cropped_header = cropped_header
        self.cropped_path = crop_output_path

        logger.info("Wrote crop: %s", crop_output_path)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            axes[0].imshow(self.image_data, origin="lower", cmap="gray")
            axes[0].set_title("Original image")
            axes[0].set_xlabel("X pixel")
            axes[0].set_ylabel("Y pixel")

            rect_x = [x0, x1, x1, x0, x0]
            rect_y = [y0, y0, y1, y1, y0]
            axes[0].plot(rect_x, rect_y, color="red", linewidth=2)

            axes[1].imshow(cropped_image, origin="lower", cmap="gray")
            axes[1].set_title("Cropped image")
            axes[1].set_xlabel("X pixel")
            axes[1].set_ylabel("Y pixel")

            plt.tight_layout()
            plt.show()

        return crop_output_path

    def circular_crop(
        self,
        crop_radius_frac: float,
        plot: bool = False,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> Path:
        """Write a circular central crop, padded outside the circle with the median."""
        if self.image_data is None or self.header is None:
            raise RuntimeError("Call open() before cropping.")

        if crop_radius_frac <= 0:
            raise ValueError("crop_radius_frac must be > 0")

        ny, nx = self.image_data.shape

        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        cx += dx
        cy += dy

        radius_pix = crop_radius_frac * min(ny, nx)
        r = float(radius_pix)

        x0 = int(np.floor(cx - r))
        x1 = int(np.ceil(cx + r))
        y0 = int(np.floor(cy - r))
        y1 = int(np.ceil(cy + r))

        x0c, x1c = max(0, x0), min(nx, x1)
        y0c, y1c = max(0, y0), min(ny, y1)

        circ_img = self.image_data[y0c:y1c, x0c:x1c].copy()
        bkg = float(np.nanmedian(circ_img))

        yy, xx = np.indices(circ_img.shape)
        cx_cut = cx - x0c
        cy_cut = cy - y0c
        mask = (xx - cx_cut) ** 2 + (yy - cy_cut) ** 2 <= r**2
        circ_img[~mask] = bkg

        h = self.header.copy()
        h["NAXIS1"] = circ_img.shape[1]
        h["NAXIS2"] = circ_img.shape[0]
        h["HIERARCH CIRC_CX"] = (float(cx), "circle centre x (orig pix)")
        h["HIERARCH CIRC_CY"] = (float(cy), "circle centre y (orig pix)")
        h["HIERARCH CIRC_R"] = (float(r), "circle radius (pix)")
        h["HIERARCH CUT_X0"] = (int(x0c), "cutout origin x (orig pix)")
        h["HIERARCH CUT_Y0"] = (int(y0c), "cutout origin y (orig pix)")

        crop_output_path = self.solved_dir / f"{self.file_path.stem}_crop.fits"
        fits.writeto(crop_output_path, circ_img, h, overwrite=True)

        self.cropped_data = circ_img
        self.cropped_header = h
        self.cropped_path = crop_output_path

        logger.info("Wrote crop: %s", crop_output_path)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            axes[0].imshow(self.image_data, origin="lower", cmap="gray")
            axes[0].set_title("Original image (solve region overlay)")
            axes[0].set_xlabel("X pixel")
            axes[0].set_ylabel("Y pixel")
            axes[0].add_patch(
                plt.Circle((cx, cy), r, edgecolor="red", facecolor="none", linewidth=2)
            )
            axes[0].plot(
                [cx - r, cx + r, cx + r, cx - r, cx - r],
                [cy - r, cy - r, cy + r, cy + r, cy - r],
                color="red",
                linestyle="--",
                linewidth=1,
            )

            axes[1].imshow(circ_img, origin="lower", cmap="gray")
            axes[1].set_title("Cutout fed to astrometry.net")
            axes[1].set_xlabel("X pixel (cutout)")
            axes[1].set_ylabel("Y pixel (cutout)")
            axes[1].add_patch(
                plt.Circle(
                    (cx_cut, cy_cut),
                    r,
                    edgecolor="red",
                    facecolor="none",
                    linewidth=2,
                )
            )

            plt.tight_layout()
            plt.show()

        return crop_output_path

    def _find_solve_field(self, solve_field_path: Optional[str] = None) -> str:
        """Find solve-field executable."""
        if solve_field_path:
            path = Path(solve_field_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"solve-field not found at: {path}")
            return str(path)

        which = shutil.which("solve-field")
        if which is not None:
            return which

        fallback_paths = [
            Path("/usr/local/bin/solve-field"),
            Path("~/.local/bin/solve-field").expanduser(),
        ]
        for path in fallback_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "Could not find solve-field. Pass --solve-field-path explicitly."
        )

    def _cleanup_astrometry_outputs(self) -> None:
        """Remove intermediate astrometry.net files for the current crop."""
        if self.cropped_path is None:
            return

        junk_exts = [".axy", ".corr", ".match", ".rdls", ".solved", ".xyls", ".wcs"]
        for ext in junk_exts:
            path = self.cropped_path.with_suffix(ext)
            if path.exists():
                path.unlink()

    def _run_solve(
        self,
        cmd: list[str],
        timeout: Optional[int] = None,
        keep_crop: bool = False,
    ) -> Path:
        """Run solve-field and return the solved FITS path."""
        if self.cropped_path is None:
            raise RuntimeError("No cropped image to solve. Call a crop method first.")

        logger.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, timeout=timeout)

        solved_new = self.cropped_path.with_suffix(".new")
        solved_fits = self.cropped_path.with_name(
            self.cropped_path.name.replace("_crop.fits", "_crop_solved.fits")
        )

        if not solved_new.exists():
            raise RuntimeError("solve-field did not produce a .new file")

        solved_new.rename(solved_fits)

        if not keep_crop and self.cropped_path.exists():
            self.cropped_path.unlink()

        self._cleanup_astrometry_outputs()
        logger.info("Solved FITS written to: %s", solved_fits)

        return solved_fits

    def solve(
        self,
        solve_field_path: Optional[str] = None,
        timeout: Optional[int] = 120,
        cpulimit: int = 40,
        index_dir: Optional[str] = None,
        keep_crop: bool = False,
    ) -> Path:
        """Run astrometry.net without an RA/Dec prior."""
        if self.cropped_path is None:
            raise RuntimeError("No cropped image to solve. Call a crop method first.")

        solve_field = self._find_solve_field(solve_field_path)

        cmd = [
            solve_field,
            str(self.cropped_path),
            "--overwrite",
            "--no-plots",
            "--guess-scale",
            "--crpix-center",
            "--cpulimit",
            str(cpulimit),
            "-v",
        ]

        if index_dir:
            cmd.extend(["--index-dir", str(Path(index_dir).expanduser())])

        return self._run_solve(cmd, timeout=timeout, keep_crop=keep_crop)

    def solve_ra_dec(
        self,
        radius_deg: float,
        solve_field_path: Optional[str] = None,
        timeout: Optional[int] = 120,
        cpulimit: int = 60,
        index_dir: Optional[str] = None,
        keep_crop: bool = False,
    ) -> Path:
        """Run astrometry.net using a zenith-based RA/Dec guess."""
        if self.cropped_path is None:
            raise RuntimeError("No cropped image to solve. Call a crop method first.")

        if self.time is None or self.location is None:
            raise RuntimeError("Call open() first to populate time and location.")

        solve_field = self._find_solve_field(solve_field_path)
        ra, dec = self.wcs_guess()

        cmd = [
            solve_field,
            str(self.cropped_path),
            "--overwrite",
            "--no-plots",
            "--guess-scale",
            "--ra",
            str(ra),
            "--dec",
            str(dec),
            "--radius",
            str(radius_deg),
            "--crpix-center",
            "--cpulimit",
            str(cpulimit),
            "-v",
        ]

        if index_dir:
            cmd.extend(["--index-dir", str(Path(index_dir).expanduser())])

        return self._run_solve(cmd, timeout=timeout, keep_crop=keep_crop)


def solve_one_file(
    fits_path: Path,
    solved_dir: Path,
    crop_mode: str,
    crop_frac: Optional[float],
    crop_radius_frac: Optional[float],
    radius_deg: Optional[float],
    use_ra_dec_guess: bool,
    plot: bool,
    dx: float,
    dy: float,
    timeout: int,
    cpulimit: int,
    solve_field_path: Optional[str],
    index_dir: Optional[str],
    keep_crop: bool,
) -> Path:
    """Solve a single FITS file and return the solved FITS path."""
    img = Solve(fits_path, solved_dir)
    img.open()

    if crop_mode == "square":
        if crop_frac is None:
            raise ValueError("--crop-frac is required for square crop mode")
        img.square_crop(crop_frac=crop_frac, plot=plot)

    elif crop_mode == "circle":
        if crop_radius_frac is None:
            raise ValueError("--crop-radius-frac is required for circle crop mode")
        img.circular_crop(
            crop_radius_frac=crop_radius_frac,
            plot=plot,
            dx=dx,
            dy=dy,
        )
    else:
        raise ValueError(f"Unknown crop mode: {crop_mode}")

    if use_ra_dec_guess:
        if radius_deg is None:
            raise ValueError("--radius-deg is required when using RA/Dec guess")
        return img.solve_ra_dec(
            radius_deg=radius_deg,
            solve_field_path=solve_field_path,
            timeout=timeout,
            cpulimit=cpulimit,
            index_dir=index_dir,
            keep_crop=keep_crop,
        )

    return img.solve(
        solve_field_path=solve_field_path,
        timeout=timeout,
        cpulimit=cpulimit,
        index_dir=index_dir,
        keep_crop=keep_crop,
    )


def batch_solve(
    raw_dir: Path,
    solved_dir: Path,
    crop_mode: str,
    crop_frac: Optional[float],
    crop_radius_frac: Optional[float],
    radius_deg: Optional[float],
    use_ra_dec_guess: bool,
    plot: bool,
    dx: float,
    dy: float,
    timeout: int,
    cpulimit: int,
    solve_field_path: Optional[str],
    index_dir: Optional[str],
    keep_crop: bool,
) -> None:
    """Batch solve all FITS files in a directory."""
    fits_files = sorted(raw_dir.glob("*.fits"))
    logger.info("Found %d FITS files in %s", len(fits_files), raw_dir)

    for i, fits_path in enumerate(fits_files, start=1):
        raw_stem = fits_path.stem
        solved_target = solved_dir / f"{raw_stem}_crop_solved.fits"

        if solved_target.exists():
            logger.info("[%d/%d] SKIP already solved: %s", i, len(fits_files), raw_stem)
            continue

        logger.info("[%d/%d] Solving: %s", i, len(fits_files), raw_stem)

        try:
            solve_one_file(
                fits_path=fits_path,
                solved_dir=solved_dir,
                crop_mode=crop_mode,
                crop_frac=crop_frac,
                crop_radius_frac=crop_radius_frac,
                radius_deg=radius_deg,
                use_ra_dec_guess=use_ra_dec_guess,
                plot=plot,
                dx=dx,
                dy=dy,
                timeout=timeout,
                cpulimit=cpulimit,
                solve_field_path=solve_field_path,
                index_dir=index_dir,
                keep_crop=keep_crop,
            )
            logger.info("SUCCESS: %s", solved_target)

        except subprocess.TimeoutExpired:
            logger.warning("TIMEOUT: %s (exceeded %ds)", raw_stem, timeout)
        except Exception as exc:
            logger.exception("FAILED: %s (%s)", raw_stem, exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop and astrometrically solve FITS all-sky images."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--crop-mode",
        choices=["square", "circle"],
        default="circle",
        help="Crop mode to use before solving.",
    )
    common.add_argument(
        "--crop-frac",
        type=float,
        default=None,
        help="Fraction removed from each side for square crop.",
    )
    common.add_argument(
        "--crop-radius-frac",
        type=float,
        default=0.17,
        help="Radius as a fraction of min(image height, image width) for circular crop.",
    )
    common.add_argument(
        "--plot",
        action="store_true",
        help="Show crop diagnostic plots.",
    )
    common.add_argument(
        "--dx",
        type=float,
        default=0.0,
        help="X shift of crop centre in pixels.",
    )
    common.add_argument(
        "--dy",
        type=float,
        default=0.0,
        help="Y shift of crop centre in pixels.",
    )
    common.add_argument(
        "--use-ra-dec-guess",
        action="store_true",
        help="Use zenith-based RA/Dec guess when solving.",
    )
    common.add_argument(
        "--radius-deg",
        type=float,
        default=60.0,
        help="Search radius in degrees for solve-field when using RA/Dec guess.",
    )
    common.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Wall-time timeout in seconds for solve-field.",
    )
    common.add_argument(
        "--cpulimit",
        type=int,
        default=60,
        help="solve-field CPU time limit in seconds.",
    )
    common.add_argument(
        "--solve-field-path",
        type=str,
        default=None,
        help="Path to solve-field executable. If omitted, auto-detect.",
    )
    common.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Optional astrometry.net index directory.",
    )
    common.add_argument(
        "--keep-crop",
        action="store_true",
        help="Keep the intermediate cropped FITS file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", parents=[common], help="Solve one FITS file.")
    single.add_argument("fits_path", type=Path, help="Path to input FITS file.")
    single.add_argument("solved_dir", type=Path, help="Directory for solved output.")

    batch = subparsers.add_parser("batch", parents=[common], help="Solve all FITS files in a directory.")
    batch.add_argument("raw_dir", type=Path, help="Directory containing FITS files.")
    batch.add_argument("solved_dir", type=Path, help="Directory for solved output.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.command == "single":
        solved_path = solve_one_file(
            fits_path=args.fits_path,
            solved_dir=args.solved_dir,
            crop_mode=args.crop_mode,
            crop_frac=args.crop_frac,
            crop_radius_frac=args.crop_radius_frac,
            radius_deg=args.radius_deg,
            use_ra_dec_guess=args.use_ra_dec_guess,
            plot=args.plot,
            dx=args.dx,
            dy=args.dy,
            timeout=args.timeout,
            cpulimit=args.cpulimit,
            solve_field_path=args.solve_field_path,
            index_dir=args.index_dir,
            keep_crop=args.keep_crop,
        )
        logger.info("Final solved file: %s", solved_path)

    elif args.command == "batch":
        batch_solve(
            raw_dir=args.raw_dir,
            solved_dir=args.solved_dir,
            crop_mode=args.crop_mode,
            crop_frac=args.crop_frac,
            crop_radius_frac=args.crop_radius_frac,
            radius_deg=args.radius_deg,
            use_ra_dec_guess=args.use_ra_dec_guess,
            plot=args.plot,
            dx=args.dx,
            dy=args.dy,
            timeout=args.timeout,
            cpulimit=args.cpulimit,
            solve_field_path=args.solve_field_path,
            index_dir=args.index_dir,
            keep_crop=args.keep_crop,
        )


if __name__ == "__main__":
    main()
