"""
Sky clarity analysis using a calibrated all-sky camera model.

This file extends the Calibrate class and uses a saved camera calibration
to map catalogue stars onto the image. It then compares expected stars
with detected stars to estimate sky clarity.

Main outputs
- Cvis: weighted star-visibility metric across HEALPix sky regions
- zero point: simple photometric quality estimate from matched stars
- diagnostic plots showing detections, matches, and sky-clarity maps

This is written as a project script, so it keeps the class structure
but stays simple and focused on the workflow actually used.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy_healpix import HEALPix
from matplotlib.colors import Normalize

from Calibration import Calibrate, ImageParams, ExtractionConfig


class SkyClarity(Calibrate):
    """
    Analyse one all-sky image using a saved calibration model.
    """

    def __init__(self, file_path, output_dir, cat_path, json_path):
        """
        Parameters
        file_path : str
            Path to the raw FITS image.
        output_dir : str
            Folder for any saved output plots.
        cat_path : str
            Path to the bright-star catalogue.
        json_path : str
            Path to the saved camera calibration JSON file.
        """
        super().__init__(file_path, solved_file=None, output_dir=output_dir, cat_path=cat_path)

        # Fixed sky mask used to avoid detections beyond the useful sky region.
        self.mask_params = [791.11, 621.99, 530.89]  # sky_xc, sky_yc, sky_r

        # Load the saved camera model from JSON.
        with open(json_path, "r") as f:
            cm = json.load(f)

        self.model = ImageParams(
            xc=float(cm["xc"]),
            yc=float(cm["yc"]),
            s=float(cm["s"]),
            theta_north_deg=float(cm["theta_north_deg"]),
            k=list(cm["k"]),
            dr=list(cm["dr"]),
            dt=list(cm["dt"]),
        )

    def stars_and_cat(self, plot=True, plot2=True, plot3=True, save=False):
        """
        Main sky-clarity workflow for a single image.

        Steps
        1. Open the image.
        2. Reject frames that are obviously too bright.
        3. Detect stars in the image.
        4. Project catalogue stars into pixel coordinates using the calibrated model.
        5. Match predicted catalogue stars to detected stars.
        6. Compute Cvis and a photometric zero point.
        7. Optionally make diagnostic plots.

        Returns
        avg_cvis : float
            Mean clarity across valid sky regions.
        avg_cvis_err : float
            Uncertainty on mean clarity.
        cvis_arr : ndarray
            Cvis values as a HEALPix array.
        zp_med : float
            Median photometric zero point.
        zp_err : float
            Uncertainty on the zero point.
        snr_med : float
            Median SNR of stars used for zero point.
        """
        self.open()
        img = self.image_data

        # Very bright frames are likely unusable for night-sky analysis.
        bright_threshold = 20000
        img_med = np.nanmedian(img)

        nside = 2
        empty_cvis = np.full(12 * nside**2, np.nan, dtype=float)

        if img_med > bright_threshold:
            print("Frame too bright for reliable night-sky analysis.")
            return np.nan, np.nan, empty_cvis, np.nan, np.nan, np.nan

        print(f"median brightness {img_med}")

        # Detect stars in the image.
        stars = self.find_stars(
            img,
            mask_params=self.mask_params,
            cfg=ExtractionConfig(minarea=8, rsn_min=3.5),
            plot=plot,
        )

        src_x = np.asarray(stars["x"], dtype=float)
        src_y = np.asarray(stars["y"], dtype=float)
        src_xy = np.column_stack((src_x, src_y))

        if len(src_xy) == 0:
            print("no detected stars")
            return np.nan, np.nan, empty_cvis, np.nan, np.nan, np.nan

        # Get the catalogue in altitude/azimuth coordinates at this observation time.
        catalog, mask_above, alt_all, az_all = self.catalog_altaz()

        # Keep only reasonably bright catalogue stars above a chosen altitude.
        alt_min = 35
        mag_max = 6.5
        max_ok = 5

        vt_all = np.asarray(catalog["vt_mag"], dtype=float)
        keep = (alt_all >= alt_min) & (vt_all <= mag_max)

        cat = catalog[keep]
        alt = np.asarray(alt_all[keep], dtype=float)
        az = np.asarray(az_all[keep], dtype=float)

        # Use the calibrated model to predict where catalogue stars should fall in the image.
        upix, vpix = self.model_altaz_to_uv(alt, az, self.model)
        cat_uv = np.column_stack((upix, vpix))

        cat_ids = np.asarray(cat["id"], dtype=str)
        id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}
        cat_vt = np.asarray(cat["vt_mag"], dtype=float)

        matches = self.match_catalog_to_sources(
            cat_uv=cat_uv,
            cat_vt=cat_vt,
            cat_id=cat_ids,
            src_xy=src_xy,
            match_radius_px=20.0,
        )

        # Extra safeguard:
        # if all matched detections are still far from their predicted positions,
        # clear the match list entirely.
        any_dist = False
        all_dist_gt = True

        for cid, vt, xobs, yobs in matches:
            i = id_to_idx.get(str(cid), None)
            if i is None:
                continue

            xpred, ypred = cat_uv[i]
            d = np.hypot(xobs - xpred, yobs - ypred)

            any_dist = True
            if d <= max_ok:
                all_dist_gt = False
                break

        if any_dist and all_dist_gt:
            matches = []

        if plot:
            img_plot = np.asarray(img, np.float32)
            finite = np.isfinite(img_plot)
            vmin, vmax = np.percentile(img_plot[finite], [1, 99.7]) if np.any(finite) else (0, 1)

            H, W = img_plot.shape

            # Plot all predicted catalogue stars.
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.imshow(img_plot, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            if len(cat_uv) > 0:
                in_frame = (
                    (cat_uv[:, 0] >= 0) & (cat_uv[:, 0] < W) &
                    (cat_uv[:, 1] >= 0) & (cat_uv[:, 1] < H)
                )
                ax1.scatter(
                    cat_uv[in_frame, 0],
                    cat_uv[in_frame, 1],
                    s=12,
                    c="dodgerblue",
                    alpha=0.45,
                    label=f"Catalog all ({in_frame.sum()})",
                )

            ax1.legend(loc="lower left")
            ax1.set_title("All catalogue stars (predicted pixel positions)")
            plt.tight_layout()
            plt.show()

            # Plot only matched stars and detections.
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_plot, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            ax.scatter(src_xy[:, 0], src_xy[:, 1], s=15, c="yellow", alpha=0.35, label="Detected")

            if len(matches) > 0:
                xpreds, ypreds, xobs_list, yobs_list = [], [], [], []

                for cid, vt, xo, yo in matches:
                    i = id_to_idx.get(str(cid))
                    if i is None:
                        continue

                    xp, yp = cat_uv[i]
                    xpreds.append(xp)
                    ypreds.append(yp)
                    xobs_list.append(xo)
                    yobs_list.append(yo)

                    ax.plot([xp, xo], [yp, yo], linewidth=1.0, alpha=0.8)

                ax.scatter(
                    xpreds,
                    ypreds,
                    s=45,
                    c="cyan",
                    alpha=0.9,
                    label=f"Catalog matched ({len(xpreds)})",
                )

                ax.scatter(
                    xobs_list,
                    yobs_list,
                    s=55,
                    facecolors="none",
                    edgecolors="lime",
                    linewidths=1.5,
                    label="Detections matched",
                )

            print(f"No. Catalog-Star Matches {len(matches)}")
            ax.legend(loc="lower left")
            ax.set_title("Matched catalogue stars (predicted → detected)")
            plt.tight_layout()
            plt.show()

        # Compute sky clarity from matched and expected stars.
        cvis, cvis_err, avg_cvis, avg_cvis_err = self.compute_Cvis(
            matches,
            cat,
            alt,
            az,
            nside=nside,
            mag_thresh=5.5,
            m0=3.5,
            s=0.35,
            min_weight_per_pix=1.3,
        )

        cvis_arr = self.cvis_dict_to_array(cvis, nside=nside)

        if plot3:
            self.plot_Cvis(cvis, nside=nside, title="Mean Sky Clarity (Cvis) by Sky Region")

        # Estimate photometric quality from matched stars.
        zp_med, zp_err, snr_med = self.zero_point(matches, cat, plot=plot)

        print("Average Cvis:", avg_cvis, avg_cvis_err)
        print("Median SNR:", snr_med)
        print("Median ZP:", zp_med)
        print("ZP error:", zp_err)

        if save:
            print("will save to file")

        # Save a plain image copy if requested.
        if plot2:
            img_plot = np.asarray(img, np.float32)
            finite = np.isfinite(img_plot)
            vmin, vmax = np.percentile(img_plot[finite], [1, 99.7]) if np.any(finite) else (0, 1)

            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.imshow(img_plot, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            ax1.axis("off")
            plt.tight_layout()

            save_path = Path(self.output_dir) / f"{Path(self.file_path).stem}_check.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()

        return avg_cvis, avg_cvis_err, cvis_arr, zp_med, zp_err, snr_med

    def zero_point(self, matches, cat, plot=True):
        """
        Compute a simple photometric zero point from matched stars.

        Only stars that pass basic magnitude, flux, and SNR cuts are used.

        Returns
        zp_med : float
            Median zero point.
        zp_err : float
            Uncertainty on the zero point from the robust scatter.
        snr_med : float
            Median SNR of the stars used.
        """
        img = np.asarray(self.image_data, float)

        cat_ids = np.asarray(cat["id"], dtype=str)
        vt_all = np.asarray(cat["vt_mag"], dtype=float)
        id_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

        zp_star = []
        star_xy = []
        snr_arr = []

        threshold_flux = 600
        min_snr = 5.0
        min_matches_zp = 10

        for cid, vt_match, xobs, yobs in matches:
            i = id_to_idx.get(str(cid))
            if i is None:
                continue

            vt = vt_all[i]
            if not np.isfinite(vt):
                continue
            if not (1.5 <= vt <= 5.5):
                continue

            flux, flux_err, snr, sky_sigma = self.aperture_flux(img, xobs, yobs)

            if not np.isfinite(flux) or not np.isfinite(flux_err) or not np.isfinite(snr):
                continue
            if flux <= threshold_flux:
                continue
            if flux <= 0:
                continue
            if snr < min_snr:
                continue

            m_inst = -2.5 * np.log10(flux)
            zp = vt - m_inst

            if not np.isfinite(zp):
                continue

            zp_star.append(zp)
            star_xy.append((xobs, yobs))
            snr_arr.append(snr)

        zp_star = np.asarray(zp_star, float)
        snr_arr = np.asarray(snr_arr, float)

        if len(zp_star) < min_matches_zp:
            print(f"Too few matched stars for reliable ZP ({len(zp_star)} < {min_matches_zp})")
            return np.nan, np.nan, np.nan

        # Robust sigma clipping.
        med = np.nanmedian(zp_star)
        mad = np.nanmedian(np.abs(zp_star - med))

        if np.isfinite(mad) and mad > 0:
            sigma_rob = 1.4826 * mad
            keep = np.abs(zp_star - med) < 3.0 * sigma_rob

            zp_star = zp_star[keep]
            snr_arr = snr_arr[keep]
            star_xy = [xy for j, xy in enumerate(star_xy) if keep[j]]

        if len(zp_star) < min_matches_zp:
            print(f"Too few matched stars after clipping for reliable ZP ({len(zp_star)} < {min_matches_zp})")
            return np.nan, np.nan, np.nan

        zp_med = np.nanmedian(zp_star)
        mad = np.nanmedian(np.abs(zp_star - zp_med))
        sigma_rob = 1.4826 * mad if np.isfinite(mad) else np.nan
        zp_err = sigma_rob / np.sqrt(len(zp_star)) if np.isfinite(sigma_rob) else np.nan

        if zp_err > 0.4:
            print("error in zp too high")
            zp_med = np.nan

        snr_med = np.nanmedian(snr_arr)

        print("Matched stars used for ZP:", len(zp_star))

        if plot:
            img_plot = np.asarray(self.image_data, np.float32)
            finite = np.isfinite(img_plot)
            vmin, vmax = np.percentile(img_plot[finite], [1, 99.7])

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_plot, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            if len(star_xy) > 0:
                star_xy = np.asarray(star_xy, float)
                ax.scatter(
                    star_xy[:, 0],
                    star_xy[:, 1],
                    s=45,
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=1.5,
                    label=f"ZP stars ({len(star_xy)})",
                )

            ax.set_title("Matched stars used for zero point")
            ax.legend(loc="lower left")
            plt.tight_layout()
            plt.show()

        return zp_med, zp_err, snr_med

    def aperture_flux(self, img, x, y):
        """
        Measure aperture flux and a simple SNR estimate for one star.

        Returns
        flux_net : float
            Background-subtracted aperture flux.
        flux_err : float
            Approximate flux uncertainty from sky noise.
        snr : float
            Signal-to-noise ratio.
        sky_sigma : float
            Standard deviation of sky pixels in the annulus.
        """
        x = float(x)
        y = float(y)

        r_ap = 5
        r_in = 8
        r_out = 10

        x0 = int(np.floor(x - r_out))
        x1 = int(np.ceil(x + r_out + 1))
        y0 = int(np.floor(y - r_out))
        y1 = int(np.ceil(y + r_out + 1))

        ny, nx = img.shape
        if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
            return np.nan, np.nan, np.nan, np.nan

        cut = img[y0:y1, x0:x1]
        if cut.size == 0:
            return np.nan, np.nan, np.nan, np.nan

        yy, xx = np.indices(cut.shape)
        xx = xx + x0
        yy = yy + y0
        rr2 = (xx - x) ** 2 + (yy - y) ** 2

        ap_mask = rr2 <= r_ap ** 2
        ann_mask = (rr2 >= r_in ** 2) & (rr2 <= r_out ** 2) & np.isfinite(cut)

        ap_pixels = cut[ap_mask]
        sky_pixels = cut[ann_mask]

        if ap_pixels.size == 0 or sky_pixels.size < 10:
            return np.nan, np.nan, np.nan, np.nan

        flux_aper = np.nansum(ap_pixels)
        bkg_mean = np.nanmedian(sky_pixels)
        flux_net = flux_aper - bkg_mean * np.sum(ap_mask)

        sky_sigma = np.nanstd(sky_pixels, ddof=1)
        if not np.isfinite(sky_sigma) or sky_sigma <= 0:
            return flux_net, np.nan, np.nan, np.nan

        flux_err = sky_sigma * np.sqrt(np.sum(ap_mask))
        snr = flux_net / flux_err if flux_err > 0 else np.nan

        return flux_net, flux_err, snr, sky_sigma

    def cvis_dict_to_array(self, cvis, nside):
        """
        Convert a dictionary of HEALPix pixel values into a full array.
        """
        npix = 12 * nside * nside
        arr = np.full(npix, np.nan, dtype=float)

        for pix, val in cvis.items():
            arr[int(pix)] = float(val)

        return arr

    def compute_Cvis(
        self,
        matches,
        cat,
        alt,
        az,
        nside=2,
        mag_thresh=6.0,
        m0=4.0,
        s=0.4,
        min_weight_per_pix=3,
        debug=False,
    ):
        """
        Compute the sky clarity metric Cvis in HEALPix sky regions.

        Cvis is the weighted fraction of expected catalogue stars that were detected.

        Returns
        cvis : dict
            Cvis value for each HEALPix pixel.
        cvis_err : dict
            Approximate uncertainty for each HEALPix pixel.
        avg_cvis : float
            Mean clarity over valid pixels.
        avg_cvis_err : float
            Uncertainty on the mean clarity.
        """
        catalog_ids = np.asarray(cat["id"]).astype(str)
        vt_mag = np.asarray(cat["vt_mag"], dtype=float)

        alt = np.asarray(alt, dtype=float)
        az = np.asarray(az, dtype=float) % 360.0

        hp = HEALPix(nside=nside, order="ring", frame=None)
        pixel_index = hp.lonlat_to_healpix(az * u.deg, alt * u.deg)
        unique_pixels = np.unique(pixel_index).astype(int)

        catalog_ok = np.isfinite(vt_mag) & (vt_mag <= mag_thresh)
        id_to_index = {catalog_ids[i]: i for i in range(len(catalog_ids))}

        # Smooth magnitude weighting: brighter stars count more strongly.
        def weight(m):
            return 1.0 / (1.0 + np.exp(-(m0 - m) / s))

        weights = weight(vt_mag)

        Wcat = {}
        for p in unique_pixels:
            sel = (pixel_index == p) & catalog_ok
            Wcat[p] = float(np.sum(weights[sel]))

        Wobs = {p: 0.0 for p in unique_pixels}
        for cid_match, vt_match, xobs, yobs in matches:
            i = id_to_index.get(str(cid_match))
            if i is None:
                continue
            if not catalog_ok[i]:
                continue

            p = int(pixel_index[i])
            Wobs[p] += float(weights[i])

        cvis = {}
        cvis_err = {}

        for p in unique_pixels:
            wcat = Wcat[p]
            wobs = Wobs.get(p, 0.0)

            if wcat < min_weight_per_pix:
                cvis[p] = np.nan
                cvis_err[p] = np.nan
                continue

            c = wobs / wcat
            c_scaled = np.clip(c, 0.0, 1.0)
            err = np.sqrt(c * (1 - c) / wcat)

            cvis[p] = float(c_scaled)
            cvis_err[p] = float(err)

        cvis_vals = np.array(list(cvis.values()), dtype=float)
        cvis_err_vals = np.array(list(cvis_err.values()), dtype=float)

        valid = np.isfinite(cvis_vals) & np.isfinite(cvis_err_vals)

        if np.any(valid):
            avg_cvis = np.nanmean(cvis_vals[valid])
            avg_cvis_err = np.sqrt(np.sum(cvis_err_vals[valid] ** 2)) / np.sum(valid)
        else:
            avg_cvis = np.nan
            avg_cvis_err = np.nan

        return cvis, cvis_err, avg_cvis, avg_cvis_err

    def plot_Cvis(
        self,
        cvis,
        nside,
        title="Cvis (equal-area polar)",
        alt_min_deg=35.0,
        n_theta=720,
        n_r=360,
        cmap="viridis",
    ):
        """
        Plot Cvis in an equal-area polar projection.

        The angle is azimuth and the radius is chosen so equal sky areas
        occupy equal plot areas.
        """
        hp = HEALPix(nside=nside, order="ring", frame=None)

        theta_edges = np.linspace(0.0, 2 * np.pi, n_theta + 1)
        r_max = np.sqrt(1.0 - np.sin(np.deg2rad(alt_min_deg)))
        r_edges = np.linspace(0.0, r_max, n_r + 1)

        theta_centres = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

        TH, RR = np.meshgrid(theta_centres, r_centres)

        az_deg = np.rad2deg(TH) % 360.0
        sin_alt = 1.0 - RR ** 2
        sin_alt = np.clip(sin_alt, 0.0, 1.0)
        alt_deg = np.rad2deg(np.arcsin(sin_alt))

        pixel_index = hp.lonlat_to_healpix(az_deg * u.deg, alt_deg * u.deg)

        values = np.full(pixel_index.shape, np.nan, dtype=float)
        for p in np.unique(pixel_index):
            values[pixel_index == p] = cvis.get(int(p), np.nan)

        TH_edges, RR_edges = np.meshgrid(theta_edges, r_edges)

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, r_max)

        values = np.ma.masked_invalid(values)

        norm = Normalize(vmin=0.0, vmax=1.0)
        ax.pcolormesh(
            TH_edges,
            RR_edges,
            values,
            cmap=cmap,
            norm=norm,
            shading="flat",
        )

        alt_ticks = np.array([20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
        alt_ticks = alt_ticks[alt_ticks >= alt_min_deg]
        r_ticks = np.sqrt(1.0 - np.sin(np.deg2rad(alt_ticks)))

        ax.set_yticks(r_ticks)
        ax.set_yticklabels([f"{int(a)}°" for a in alt_ticks], color="black", fontsize=19)
        ax.set_rlabel_position(70)
        ax.tick_params(axis="y", colors="black")

        az_ticks_deg = np.arange(0, 360, 45)
        ax.set_xticks(np.deg2rad(az_ticks_deg))
        ax.set_xticklabels([f"{d}°" for d in az_ticks_deg], fontsize=19)

        ax.text(
            0.5,
            1.08,
            "Azimuth (°)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=22,
        )

        ax.text(
            0.78,
            0.44,
            "Altitude (°)",
            transform=ax.transAxes,
            color="black",
            rotation=-70,
            ha="center",
            va="center",
            fontsize=22,
        )

        plt.tight_layout()

        save_path = Path(self.output_dir) / f"{Path(self.file_path).stem}_cvis.png"
        # plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


# Example usage
fits_file = "/path/to/raw_image.fits"
output_path = "/path/to/output_directory"
catalog_path = "/path/to/catalog_file.fits"
json_path = "/path/to/camera_calibration.json"

s = SkyClarity(fits_file, output_path, catalog_path, json_path)
result = s.stars_and_cat(plot=False, plot2=True, plot3=True)
