"""
Camera calibration for the all-sky images.

This file is used to calibrate the mapping between sky coordinates
(altitude, azimuth) and image pixel coordinates (u, v).

Main idea

1. Open one raw FITS image and read its observing time/location.
2. Use a solved cropped image to get an initial set of reliable WCS matches.
3. Use those matches to fit a camera model.
4. Optionally refine the model using detections across the full image.
5. Save or inspect the fitted calibration parameters.

This is written as a project script, so it is kept simple and focused
on the workflow actually used in the dissertation.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import sep

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.table import Table

from scipy.optimize import least_squares
from scipy.spatial import cKDTree


class ImageParams:
    """
    Store the camera model parameters.

    Parameters
    xc, yc : float
        Pixel coordinates of the zenith.
    s : float
        Zenith-to-horizon scale in pixels.
    theta_north_deg : float
        Rotation angle giving the north direction in the image.
    k : list
        Projection polynomial coefficients.
    dr : list
        Radial distortion coefficients.
    dt : list
        Tangential distortion coefficients.
    """

    def __init__(self, xc, yc, s, theta_north_deg, k, dr, dt):
        self.xc = xc
        self.yc = yc
        self.s = s
        self.theta_north_deg = theta_north_deg
        self.k = k
        self.dr = dr
        self.dt = dt


@dataclass
class ExtractionConfig:
    """
    Settings used for SEP source extraction and catalogue matching.
    """

    # Background estimation settings
    bw: int = 64
    bh: int = 64
    fw: int = 5
    fh: int = 5

    # Detection settings
    thresh_sigma: float = 1.6
    minarea: int = 8
    maxarea: int = 200

    # Screening settings
    rsn_min: float = 3.0
    tmetric_max: float = 7.0
    rsn_bright: float = 10.0
    a_max: float = 10.0

    # Matching settings
    match_radius_px: float = 15.0

    # Sky cuts
    nside: int = 8
    alt_min_deg: float = 25.0


class Calibrate:
    """
    Handle image loading, star detection, WCS matching, and camera calibration.
    """

    def __init__(self, file_path, solved_file, output_dir, cat_path):
        self.file_path = Path(file_path)
        self.solved_file = Path(solved_file)
        self.output_dir = Path(output_dir)
        self.cat_path = Path(cat_path)

        self.header = None
        self.image_data = None

        # Filled when the raw FITS is opened
        self.obstime = None
        self.location = None
        self.elevation = None

    def open(self):
        """
        Open the raw FITS image and read observation metadata.

        Reads:
        - image data
        - header
        - DATE-OBS
        - LONGITUD / LONGITUDE
        - LATITUDE
        - ALTITUDE
        """
        with fits.open(self.file_path) as hdul:
            self.header = hdul[0].header.copy()
            self.image_data = hdul[0].data

        date_time = self.header.get("DATE-OBS")
        lon = self.header.get("LONGITUD", self.header.get("LONGITUDE"))
        lat = self.header.get("LATITUDE")
        height_m = self.header.get("ALTITUDE", 0) * 1000  # assumes header value is in km

        self.obstime = Time(date_time, format="isot", scale="utc")
        self.location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height_m * u.m)
        self.elevation = float(height_m)

        return self.obstime, self.location, self.elevation

    def load_wcs_and_offsets(self):
        """
        Open the solved cropped FITS file.

        Returns
        w : astropy.wcs.WCS
            WCS solution for the cropped image.
        (x0, y0) : tuple
            Pixel offset of the crop relative to the full image.
        data : ndarray
            Cropped image data.
        hdr : fits.Header
            Header of the solved cropped file.
        """
        with fits.open(self.solved_file) as hdul:
            hdr = hdul[0].header
            w = WCS(hdr)
            data = hdul[0].data

        x0 = int(hdr.get("HIERARCH CUT_X0", 0))
        y0 = int(hdr.get("HIERARCH CUT_Y0", 0))

        return w, (x0, y0), data, hdr

    def wcs_guess(self):
        """
        Estimate the sky coordinates of the zenith.

        This is mainly useful as a quick check that the image metadata
        gives sensible sky coordinates.
        """
        zenith_altaz = SkyCoord(
            alt=90 * u.deg,
            az=0 * u.deg,
            frame=AltAz(obstime=self.obstime, location=self.location),
        )
        zenith_icrs = zenith_altaz.icrs
        return zenith_icrs.ra.deg, zenith_icrs.dec.deg

    @staticmethod
    def circular_mask(shape, xc, yc, r, margin=0):
        """
        Build a circular mask for SEP.

        Returns
        mask : ndarray of bool
            True outside the sky region, False inside.
        """
        yy, xx = np.indices(shape)
        rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        return rr > (r - margin)

    def find_stars(
        self,
        img,
        mask_params,
        cfg=ExtractionConfig(),
        return_bkg=False,
        plot=False,
        fov_margin_px=10,
    ):
        """
        Detect stars using SEP and apply simple filtering.

        Steps
        1. Mask pixels outside the sky region.
        2. Estimate the spatially varying background.
        3. Detect sources above threshold.
        4. Filter detections using significance, size, and shape.
        """
        img = np.asarray(img, dtype=np.float32)
        sky_xc, sky_yc, sky_r = mask_params

        mask = self.circular_mask(img.shape, sky_xc, sky_yc, sky_r, margin=fov_margin_px)

        bkg = sep.Background(img, mask=mask, bw=cfg.bw, bh=cfg.bh, fw=cfg.fw, fh=cfg.fh)
        bkg_img = bkg.back()
        rms_img = bkg.rms()

        data_sub = img - bkg_img
        thresh = cfg.thresh_sigma * bkg.globalrms

        valid = ~mask
        above = (data_sub > thresh) & valid
        frac = above.sum() / valid.sum()

        # If too much of the image is above threshold, treat the frame as bad.
        if frac > 0.50:
            stars = Table(names=["x", "y", "flux", "a", "b", "theta", "npix"], dtype=[float] * 7)
            return (stars, bkg_img, rms_img) if return_bkg else stars

        try:
            sources = sep.extract(
                data_sub,
                thresh=thresh,
                minarea=cfg.minarea,
                mask=mask,
                deblend_nthresh=32,
                deblend_cont=0.0005,
                clean=True,
                clean_param=1.0,
            )
        except Exception as e:
            if "internal pixel buffer full" in str(e):
                stars = Table(names=["x", "y", "flux", "a", "b", "theta", "npix"], dtype=[float] * 7)
                return (stars, bkg_img, rms_img) if return_bkg else stars
            raise

        x0 = np.clip(np.rint(sources["x"]).astype(int), 0, img.shape[1] - 1)
        y0 = np.clip(np.rint(sources["y"]).astype(int), 0, img.shape[0] - 1)

        bkg_at = bkg_img[y0, x0]
        rms_at = rms_img[y0, x0]
        rms_at = np.where(rms_at > 0, rms_at, np.nan)

        # Estimate a local peak near each centroid.
        peak = np.empty(len(x0), dtype=float)
        half = 2
        H, W = img.shape

        for i, (xi, yi) in enumerate(zip(x0, y0)):
            x1 = max(0, xi - half)
            x2 = min(W, xi + half + 1)
            y1 = max(0, yi - half)
            y2 = min(H, yi + half + 1)
            peak[i] = np.nanmax(img[y1:y2, x1:x2])

        rsn = (peak - bkg_at) / rms_at
        npix = sources["npix"].astype(float)

        tmetric = np.full_like(rsn, np.inf, dtype=float)
        good_rsn = np.isfinite(rsn) & (rsn > 0)
        tmetric[good_rsn] = npix[good_rsn] / rsn[good_rsn]

        a = sources["a"]

        keep = (
            np.isfinite(rsn)
            & (rsn > cfg.rsn_min)
            & np.isfinite(a)
            & ((((npix <= cfg.maxarea) & (a <= cfg.a_max) & (tmetric < cfg.tmetric_max))) | (rsn > cfg.rsn_bright))
        )

        sources = sources[keep]

        if len(sources) == 0:
            stars = Table(names=["x", "y", "flux", "a", "b", "theta", "npix"], dtype=[float] * 7)
            return (stars, bkg_img, rms_img) if return_bkg else stars

        stars = Table()
        stars["x"] = sources["x"]
        stars["y"] = sources["y"]
        stars["flux"] = sources["flux"]
        stars["a"] = sources["a"]
        stars["b"] = sources["b"]
        stars["theta"] = sources["theta"]
        stars["npix"] = sources["npix"]

        if plot:
            finite = np.isfinite(img)
            vmin, vmax = np.percentile(img[finite], [1, 99.7]) if np.any(finite) else (0, 1)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            ax.scatter(stars["x"], stars["y"], s=12, facecolors="none", edgecolors="lime", linewidths=0.8)
            ax.set_title(f"SEP detections after screening: N={len(stars)}")
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("y [pix]")
            plt.tight_layout()
            plt.show()

        return (stars, bkg_img, rms_img) if return_bkg else stars

    def load_catalog(self):
        """
        Load the bright-star catalogue.

        Expected output fields
        id, ra_deg, dec_deg, vt_mag
        """
        table = Table.read(self.cat_path)

        ra_deg = np.array(table["_RA"], dtype=float)
        dec_deg = np.array(table["_DE"], dtype=float)
        vt_mag = np.array(table["Vmag"], dtype=float)
        ids = np.array(table["LID"]).astype(str)

        rec = np.core.records.fromarrays(
            [ids, ra_deg, dec_deg, vt_mag],
            names=["id", "ra_deg", "dec_deg", "vt_mag"],
        )
        return rec

    def catalog_altaz(self):
        """
        Convert catalogue stars from RA/Dec to altitude/azimuth
        at the image observation time and location.
        """
        catalog = self.load_catalog()
        sc = SkyCoord(ra=catalog["ra_deg"] * u.deg, dec=catalog["dec_deg"] * u.deg, frame="icrs")
        altaz = sc.transform_to(AltAz(obstime=self.obstime, location=self.location))

        alt = altaz.alt.deg
        az = altaz.az.deg
        mask = alt > 0.0

        return catalog, mask, alt, az

    def match_using_wcs(self, cfg=ExtractionConfig(), plot=False):
        """
        Use the WCS solution on the solved crop to produce reliable initial matches.

        This gives a good starting set of catalogue-star / detected-star pairs
        before fitting the full camera model.
        """
        w, (x0, y0), crop_img, hdr = self.load_wcs_and_offsets()

        cx_cut = float(hdr["HIERARCH CIRC_CX"]) - float(hdr["HIERARCH CUT_X0"])
        cy_cut = float(hdr["HIERARCH CIRC_CY"]) - float(hdr["HIERARCH CUT_Y0"])
        r = float(hdr["HIERARCH CIRC_R"])
        mask_params_crop = [cx_cut, cy_cut, r]

        stars_crop = self.find_stars(img=crop_img, mask_params=mask_params_crop, cfg=cfg, plot=plot)
        if len(stars_crop) == 0:
            return [], self.load_catalog()[0:0]

        src_xy_crop = np.vstack([stars_crop["x"], stars_crop["y"]]).T.astype(float)

        catalog = self.load_catalog()
        sc = SkyCoord(ra=catalog["ra_deg"] * u.deg, dec=catalog["dec_deg"] * u.deg, frame="icrs")
        x_pred, y_pred = w.world_to_pixel(sc)
        cat_xy_crop = np.vstack([x_pred, y_pred]).T.astype(float)

        finite = np.isfinite(cat_xy_crop[:, 0]) & np.isfinite(cat_xy_crop[:, 1])
        dx = cat_xy_crop[:, 0] - cx_cut
        dy = cat_xy_crop[:, 1] - cy_cut
        inside_circle = (dx * dx + dy * dy) <= (r * r)

        keep = finite & inside_circle
        cat2 = catalog[keep]
        cat_xy_crop2 = cat_xy_crop[keep]
        vt2 = cat2["vt_mag"].astype(float)

        recognized_crop = self.match_mnn(
            cat_xy_crop2, vt2, cat2["id"], src_xy_crop, match_radius_px=cfg.match_radius_px
        )

        recognized_full = [(cid, vt, x + x0, y + y0) for (cid, vt, x, y) in recognized_crop]

        return recognized_full, cat2

    def model_altaz_to_uv(self, alt_deg, az_deg, model):
        """
        Camera model mapping altitude/azimuth to image pixel coordinates.
        """

        def poly_rho_to_r(rho, k):
            r = np.zeros_like(rho, dtype=float)
            for ki, p in zip(k, [1, 3, 5, 7, 9]):
                r += ki * np.power(rho, p)
            return r

        def delta_terms(rho_n, theta, d):
            d1, d2, d3, d4, d5, d6, d7 = d
            A = d1 * rho_n + d2 * rho_n**3 + d3 * rho_n**5
            B = d4 * np.cos(theta) + d5 * np.sin(theta) + d6 * np.cos(2 * theta) + d7 * np.sin(2 * theta)
            return A * B

        alt = np.deg2rad(alt_deg)
        az = np.deg2rad(az_deg)

        rho = model.s * (1.0 - (2.0 * alt / np.pi))
        rho_n = rho / model.s

        theta = az - np.deg2rad(model.theta_north_deg)

        r = poly_rho_to_r(rho, np.asarray(model.k, dtype=float))
        delta_r = delta_terms(rho_n, theta, np.asarray(model.dr, dtype=float))
        delta_t = delta_terms(rho_n, theta, np.asarray(model.dt, dtype=float))

        r_p = r + delta_r
        t_p = theta + delta_t

        x = r_p * np.cos(t_p)
        y = r_p * np.sin(t_p)

        u = x + model.xc
        v = y + model.yc
        return u, v

    @staticmethod
    def match_mnn(cat_uv, cat_vt, cat_id, src_xy, match_radius_px=6.0):
        """
        Mutual nearest-neighbour matching between catalogue predictions and detections.

        Returns
        list of tuples
            (catalogue id, magnitude, x_obs, y_obs)
        """
        if len(cat_uv) == 0 or len(src_xy) == 0:
            return []

        cat_uv = np.asarray(cat_uv, float)
        src_xy = np.asarray(src_xy, float)

        tree_src = cKDTree(src_xy)
        tree_cat = cKDTree(cat_uv)

        d_cs, j_src = tree_src.query(cat_uv, k=1)
        d_sc, i_cat = tree_cat.query(src_xy, k=1)

        cat_idx = np.arange(len(cat_uv))
        mutual = (i_cat[j_src] == cat_idx) & (d_cs < match_radius_px)

        cand_i = np.where(mutual)[0]
        if cand_i.size == 0:
            return []

        order = np.argsort(d_cs[cand_i])
        used_src = set()
        out = []

        for ii in cand_i[order]:
            jj = int(j_src[ii])
            if jj in used_src:
                continue
            used_src.add(jj)
            out.append((str(cat_id[ii]), float(cat_vt[ii]), float(src_xy[jj, 0]), float(src_xy[jj, 1])))

        return out

    @staticmethod
    def _pack_model(model):
        """
        Convert ImageParams into a flat parameter vector for least-squares fitting.
        """
        return np.array(
            [model.s, model.theta_north_deg, model.xc, model.yc] + list(model.k) + list(model.dr) + list(model.dt),
            dtype=float,
        )

    @staticmethod
    def _unpack_model(x):
        """
        Convert a flat least-squares vector back into ImageParams.
        """
        x = np.asarray(x, dtype=float)
        s, theta_n, xc, yc = x[0:4]
        k = x[4:9].tolist()
        dr = x[9:16].tolist()
        dt = x[16:23].tolist()
        return ImageParams(xc, yc, s, theta_n, k, dr, dt)

    def _residuals_uv(self, x, alt_deg, az_deg, uv_obs):
        """
        Residual vector used in least-squares fitting.
        """
        model = self._unpack_model(x)
        u_pred, v_pred = self.model_altaz_to_uv(alt_deg, az_deg, model)
        uv_pred = np.vstack([u_pred, v_pred]).T
        return (uv_pred - uv_obs).ravel()

    def _fit(self, x0, mask_free, alt_deg, az_deg, uv_obs, robust=True):
        """
        Fit only the selected model parameters.

        mask_free says which entries in the parameter vector are allowed to vary.
        """
        x0 = np.asarray(x0, float)
        mask_free = np.asarray(mask_free, bool)

        def fun(p_free):
            x = x0.copy()
            x[mask_free] = p_free
            return self._residuals_uv(x, alt_deg, az_deg, uv_obs)

        p0 = x0[mask_free]
        loss = "soft_l1" if robust else "linear"
        res = least_squares(fun, p0, loss=loss, f_scale=3.0, max_nfev=2000)

        x_out = x0.copy()
        x_out[mask_free] = res.x
        return x_out, res

    def calibrate_using_wcs(self, init_model, cfg=ExtractionConfig(), min_matches=15, staged=True, plot=False):
        """
        Fit the camera model using only WCS-seeded matches.

        This is usually the safest first-stage calibration.
        """
        if self.image_data is None or self.obstime is None or self.location is None:
            self.open()

        recognized_full, cat2 = self.match_using_wcs(cfg=cfg, plot=plot)

        if len(recognized_full) < min_matches:
            raise RuntimeError(f"Too few WCS matches ({len(recognized_full)}).")

        id_to_idx = {str(cid): i for i, cid in enumerate(cat2["id"])}

        alt_m, az_m, uv_obs = [], [], []
        for cid, _, xobs, yobs in recognized_full:
            i = id_to_idx.get(str(cid))
            if i is None:
                continue

            ra = float(cat2["ra_deg"][i])
            dec = float(cat2["dec_deg"][i])

            sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            altaz = sc.transform_to(AltAz(obstime=self.obstime, location=self.location))

            alt_m.append(float(altaz.alt.deg))
            az_m.append(float(altaz.az.deg))
            uv_obs.append([float(xobs), float(yobs)])

        alt_m = np.asarray(alt_m, dtype=float)
        az_m = np.asarray(az_m, dtype=float)
        uv_obs = np.asarray(uv_obs, dtype=float)

        x = self._pack_model(init_model)

        if staged:
            mask_k = np.zeros_like(x, bool)
            mask_k[4:9] = True
            x, _ = self._fit(x, mask_k, alt_m, az_m, uv_obs)

            mask_d = np.zeros_like(x, bool)
            mask_d[9:23] = True
            x, _ = self._fit(x, mask_d, alt_m, az_m, uv_obs)

            mask_g = np.zeros_like(x, bool)
            mask_g[0:4] = True
            x, _ = self._fit(x, mask_g, alt_m, az_m, uv_obs)

            mask_all = np.ones_like(x, bool)
            x, final_res = self._fit(x, mask_all, alt_m, az_m, uv_obs)
        else:
            mask_all = np.ones_like(x, bool)
            x, final_res = self._fit(x, mask_all, alt_m, az_m, uv_obs)

        best_model = self._unpack_model(x)

        rms_px = float(np.sqrt(np.mean(final_res.fun**2)))
        diag = {
            "n_wcs_matches": int(len(recognized_full)),
            "n_used_in_fit": int(len(alt_m)),
            "rms_px": rms_px,
            "success": bool(final_res.success),
            "message": str(final_res.message),
        }

        return best_model, diag

    def calibrate_two_stage(self, init_model, cfg_wcs, cfg_full, plot=False):
        """
        Two-stage calibration.

        Stage A:
            Fit using WCS-seeded matches from the solved crop.

        Stage B:
            Refine further using the full-image calibration routine.
        """
        model_seed, diag_seed = self.calibrate_using_wcs(
            init_model=init_model,
            cfg=cfg_wcs,
            staged=True,
            plot=False,
        )

        model_full, diag_full = self.calibrate(
            init_model=model_seed,
            cfg=cfg_full,
            staged=True,
            plot=plot,
        )

        return model_full, {"seed": diag_seed, "full": diag_full}

    def calibrate(self, init_model, cfg=ExtractionConfig(), min_matches=15, staged=True, plot=False):
        """
        Full-image calibration.

        Starts from an initial camera model, predicts catalogue-star positions,
        matches them to observed detections, and then refits the model.
        """
        if self.image_data is None or self.obstime is None or self.location is None:
            self.open()

        # Fixed sky mask values used for the full image
        sky_xc = 791.11
        sky_yc = 621.99
        sky_r = 530.89
        mask_params = [sky_xc, sky_yc, sky_r]

        stars = self.find_stars(img=self.image_data, mask_params=mask_params, cfg=cfg, plot=plot)
        if len(stars) == 0:
            raise RuntimeError("No stars detected after screening.")

        src_xy = np.vstack([stars["x"], stars["y"]]).T.astype(float)

        catalog, mask, alt_all, az_all = self.catalog_altaz()
        cat = catalog[mask]
        alt = alt_all[mask].astype(float)
        az = az_all[mask].astype(float)
        vt = cat["vt_mag"].astype(float)

        u0, v0 = self.model_altaz_to_uv(alt, az, init_model)
        cat_uv0 = np.vstack([u0, v0]).T

        recognized = self.match_mnn(cat_uv0, vt, cat["id"], src_xy, match_radius_px=cfg.match_radius_px)

        if len(recognized) < min_matches:
            raise RuntimeError(f"Too few matches ({len(recognized)}).")

        id_to_idx = {str(cid): i for i, cid in enumerate(cat["id"])}

        alt_m, az_m, uv_obs = [], [], []
        for cid, _, uobs, vobs in recognized:
            i = id_to_idx.get(str(cid))
            if i is None:
                continue
            alt_m.append(alt[i])
            az_m.append(az[i])
            uv_obs.append([uobs, vobs])

        alt_m = np.asarray(alt_m, float)
        az_m = np.asarray(az_m, float)
        uv_obs = np.asarray(uv_obs, float)

        x = self._pack_model(init_model)

        if staged:
            mask_k = np.zeros_like(x, bool)
            mask_k[4:9] = True
            x, _ = self._fit(x, mask_k, alt_m, az_m, uv_obs)

            mask_d = np.zeros_like(x, bool)
            mask_d[9:23] = True
            x, _ = self._fit(x, mask_d, alt_m, az_m, uv_obs)

            mask_g = np.zeros_like(x, bool)
            mask_g[0:4] = True
            x, _ = self._fit(x, mask_g, alt_m, az_m, uv_obs)

            mask_all = np.ones_like(x, bool)
            x, final_res = self._fit(x, mask_all, alt_m, az_m, uv_obs)
        else:
            mask_all = np.ones_like(x, bool)
            x, final_res = self._fit(x, mask_all, alt_m, az_m, uv_obs)

        best_model = self._unpack_model(x)

        rms_px = float(np.sqrt(np.mean(final_res.fun**2)))
        diag = {
            "n_detected": int(len(stars)),
            "n_matched": int(len(alt_m)),
            "rms_px": rms_px,
            "success": bool(final_res.success),
            "message": str(final_res.message),
        }

        if plot:
            img = np.asarray(self.image_data, dtype=np.float32)
            finite = np.isfinite(img)
            vmin, vmax = np.percentile(img[finite], [1, 99.7]) if np.any(finite) else (0, 1)

            u_fit, v_fit = self.model_altaz_to_uv(alt_m, az_m, best_model)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            ax.scatter(
                uv_obs[:, 0],
                uv_obs[:, 1],
                s=30,
                facecolors="none",
                edgecolors="lime",
                linewidths=1.0,
                label="Observed",
            )
            ax.scatter(u_fit, v_fit, s=18, c="yellow", label="Predicted")
            ax.set_title(f"Overlay: matched stars (RMS={rms_px:.2f}px)")
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("y [pix]")
            ax.legend(loc="lower left")
            plt.tight_layout()
            plt.show()

        return best_model, diag

    def save_model(self, model, path):
        """
        Save a fitted model to JSON.
        """
        payload = {
            "xc": float(model.xc),
            "yc": float(model.yc),
            "s": float(model.s),
            "theta_north_deg": float(model.theta_north_deg),
            "k": [float(v) for v in model.k],
            "dr": [float(v) for v in model.dr],
            "dt": [float(v) for v in model.dt],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def plot_altitude_map(self, model_path, sky_mask_params=(791.11, 621.99, 530.89), step=4, plot_on_image=False):
        """
        Plot altitude across image coordinates using a saved calibration model.

        This is a useful visual check that altitude varies sensibly across the image.
        """
        with open(model_path, "r") as f:
            calib = json.load(f)

        model = SimpleNamespace(**calib)

        if self.image_data is None:
            self.open()

        img = np.asarray(self.image_data, dtype=np.float32)
        H, W = img.shape

        sky_xc, sky_yc, sky_r = map(float, sky_mask_params)

        yy, xx = np.mgrid[0:H:step, 0:W:step]
        xx = xx.astype(float)
        yy = yy.astype(float)

        r_pix = np.sqrt((xx - model.xc) ** 2 + (yy - model.yc) ** 2)
        inside = ((xx - sky_xc) ** 2 + (yy - sky_yc) ** 2) <= (sky_r ** 2)

        k = np.asarray(model.k, dtype=float)

        def r_from_rho(rho):
            out = np.zeros_like(rho, dtype=float)
            for ki, p in zip(k, [1, 3, 5, 7, 9]):
                out += ki * rho**p
            return out

        rho_lo = np.zeros_like(r_pix, dtype=float)
        rho_hi = np.full_like(r_pix, float(model.s), dtype=float)

        for _ in range(30):
            rho_mid = 0.5 * (rho_lo + rho_hi)
            r_mid = r_from_rho(rho_mid)
            go_hi = r_mid > r_pix
            rho_hi[go_hi] = rho_mid[go_hi]
            rho_lo[~go_hi] = rho_mid[~go_hi]

        rho_est = 0.5 * (rho_lo + rho_hi)
        alt_rad = (np.pi / 2.0) * (1.0 - rho_est / model.s)
        alt_deg = np.rad2deg(alt_rad)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")

        alt_masked = ma.array(alt_deg, mask=~inside)
        alt_masked = ma.masked_invalid(alt_masked)
        alt_masked = ma.clip(alt_masked, 0.0, 90.0)

        if plot_on_image:
            ax.imshow(img, origin="lower", cmap="gray")

        cmap = plt.cm.turbo.copy()
        cmap.set_bad(color="white")

        im = ax.imshow(
            alt_masked,
            origin="lower",
            extent=[0, W, 0, H],
            interpolation="nearest",
            cmap=cmap,
            alpha=0.8 if plot_on_image else 1.0,
        )

        ax.scatter([model.xc], [model.yc], s=30, facecolor="none", edgecolor="black", marker="o", label="Zenith")
        ax.add_patch(plt.Circle((sky_xc, sky_yc), sky_r + 1, edgecolor="white", facecolor="none", linewidth=3.0))

        ax.set_title("Altitude map in image coordinates")
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label("Altitude [deg]")
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.show()


# Example usage
target_path = "/path/to/raw_fits_image.fits"
solved_path = "/path/to/solved_crop_fits_image.fits"
output_path = "/path/to/output_directory"
catalog_path = "/path/to/bright_star_catalog.fits"

solver = Calibrate(target_path, solved_path, output_path, catalog_path)
solver.open()

init = ImageParams(
    xc=758.19229,
    yc=631.70484,
    s=496.57,
    theta_north_deg=200.0,
    k=[1, 0, 1, 0, 0],
    dr=[1e-5, 0.0, 0.0, 1e-5, 0.0, 0.0, 0.0],
    dt=[1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
)

cfg_wcs = ExtractionConfig(match_radius_px=5.0)
cfg_full = ExtractionConfig(match_radius_px=10.0)

best_model, diag = solver.calibrate_two_stage(
    init_model=init,
    cfg_wcs=cfg_wcs,
    cfg_full=cfg_full,
    plot=True,
)

print(diag)
print(best_model.xc, best_model.yc, best_model.s, best_model.theta_north_deg)
