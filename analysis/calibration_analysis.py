from Calibration import Calibrate, ImageParams, ExtractionConfig

import json
import numpy as np
import matplotlib.pyplot as plt


class CalibrationDiagnostics(Calibrate):
    def __init__(
        self,
        file_path: str,
        output_dir: str,
        cat_path: str,
        json_path: str = "/Users/samcavens/Documents/4th/SHP/fits_playground/camera_calibration.json",
    ):
        super().__init__(
            file_path=file_path,
            solved_file=None,
            output_dir=output_dir,
            cat_path=cat_path,
        )
        self.json_path = json_path
        self.model = self.load_model(json_path)

    def load_model(self, path: str) -> ImageParams:
        with open(path, "r") as f:
            cm = json.load(f)

        return ImageParams(
            xc=float(cm["xc"]),
            yc=float(cm["yc"]),
            s=float(cm["s"]),
            theta_north_deg=float(cm["theta_north_deg"]),
            k=list(cm["k"]),
            dr=list(cm["dr"]),
            dt=list(cm["dt"]),
        )

    def get_matched_star_residuals(
        self,
        cfg: ExtractionConfig = ExtractionConfig(),
        alt_min_deg: float = 0.0,
    ):
        """
        Returns matched-star residual diagnostics using the loaded camera model.
        """
        if self.image_data is None or self.obstime is None or self.location is None:
            self.open()

        sky_xc = 791.11
        sky_yc = 621.99
        sky_r = 530.89
        mask_params = [sky_xc, sky_yc, sky_r]

        stars = self.find_stars(
            img=self.image_data,
            mask_params=mask_params,
            cfg=cfg,
            plot=False,
        )
        if len(stars) == 0:
            raise RuntimeError("No stars detected.")

        src_xy = np.vstack([stars["x"], stars["y"]]).T.astype(float)

        catalog, mask, alt_all, az_all = self.catalog_altaz()
        cat = catalog[mask]
        alt = alt_all[mask].astype(float)
        az = az_all[mask].astype(float)
        vt = cat["vt_mag"].astype(float)

        keep_alt = alt >= alt_min_deg
        cat = cat[keep_alt]
        alt = alt[keep_alt]
        az = az[keep_alt]
        vt = vt[keep_alt]

        if len(cat) == 0:
            raise RuntimeError(
                f"No catalogue stars remain above alt_min_deg={alt_min_deg:.1f} deg."
            )

        u0, v0 = self.model_altaz_to_uv(alt, az, self.model)
        cat_uv0 = np.vstack([u0, v0]).T

        recognized = self.match_mnn(
            cat_uv0,
            vt,
            cat["id"],
            src_xy,
            match_radius_px=cfg.match_radius_px,
        )
        if len(recognized) == 0:
            raise RuntimeError("No model-based matches found.")

        id_to_idx = {str(cid): i for i, cid in enumerate(cat["id"])}

        alt_m, az_m, uv_obs = [], [], []
        for cid, vt_mag, xobs, yobs in recognized:
            i = id_to_idx.get(str(cid))
            if i is None:
                continue
            alt_m.append(float(alt[i]))
            az_m.append(float(az[i]))
            uv_obs.append([float(xobs), float(yobs)])

        alt_m = np.asarray(alt_m, dtype=float)
        az_m = np.asarray(az_m, dtype=float)
        uv_obs = np.asarray(uv_obs, dtype=float)

        if len(uv_obs) == 0:
            raise RuntimeError("No matched stars available for residual analysis.")

        u_pred, v_pred = self.model_altaz_to_uv(alt_m, az_m, self.model)
        uv_pred = np.vstack([u_pred, v_pred]).T.astype(float)

        du_px = uv_pred[:, 0] - uv_obs[:, 0]
        dv_px = uv_pred[:, 1] - uv_obs[:, 1]
        residual_px = np.hypot(du_px, dv_px)

        radius_px = np.hypot(
            uv_obs[:, 0] - self.model.xc,
            uv_obs[:, 1] - self.model.yc,
        )

        return radius_px, residual_px, du_px, dv_px, uv_obs, uv_pred, alt_m

    def plot_residual_vs_radius(
        self,
        cfg: ExtractionConfig = ExtractionConfig(),
        alt_min_deg: float = 0.0,
        bin_width_px: float = 25.0,
        min_per_bin: int = 5,
        save_path: str = None,
    ):
        """
        Plot per-star positional residual against radial distance from fitted centre.
        """
        (
            radius_px,
            residual_px,
            du_px,
            dv_px,
            uv_obs,
            uv_pred,
            alt_m,
        ) = self.get_matched_star_residuals(
            cfg=cfg,
            alt_min_deg=alt_min_deg,
        )

        overall_rmse_px = float(np.sqrt(np.mean(residual_px**2)))

        rmax = float(np.nanmax(radius_px))
        edges = np.arange(0.0, rmax + bin_width_px, bin_width_px)
        if len(edges) < 2:
            edges = np.array([0.0, rmax + 1.0])

        bin_centres = 0.5 * (edges[:-1] + edges[1:])
        bin_rmse = np.full(len(bin_centres), np.nan, dtype=float)
        bin_counts = np.zeros(len(bin_centres), dtype=int)

        for i in range(len(bin_centres)):
            in_bin = (radius_px >= edges[i]) & (radius_px < edges[i + 1])
            n = int(np.sum(in_bin))
            bin_counts[i] = n
            if n >= min_per_bin:
                e = residual_px[in_bin]
                bin_rmse[i] = np.sqrt(np.mean(e**2))

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(
            radius_px,
            residual_px,
            s=10,
            alpha=0.15,
            label="Matched stars",
        )

        good = np.isfinite(bin_rmse)
        ax.plot(
            bin_centres[good],
            bin_rmse[good],
            marker="o",
            linewidth=2,
            label=f"Binned RMSE (overall = {overall_rmse_px:.2f} px)",
        )

        ax.set_xlabel("Radius from centre [px]", size=12)
        ax.set_ylabel("Residual [px]", size=12)
        ax.legend()

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

        return {
            "radius_px": radius_px,
            "residual_px": residual_px,
            "du_px": du_px,
            "dv_px": dv_px,
            "u_obs": uv_obs[:, 0],
            "v_obs": uv_obs[:, 1],
            "u_pred": uv_pred[:, 0],
            "v_pred": uv_pred[:, 1],
            "alt_matched_deg": alt_m,
            "bin_centres_px": bin_centres,
            "bin_rmse_px": bin_rmse,
            "bin_counts": bin_counts,
            "overall_rmse_px": overall_rmse_px,
            "n_matches": len(residual_px),
            "alt_min_deg": alt_min_deg,
        }

    def plot_catalog_overlay_zoom(
        self,
        cfg: ExtractionConfig = ExtractionConfig(),
        alt_min_deg: float = 0.0,
        xlim: tuple = None,
        ylim: tuple = None,
        centre: tuple = None,
        half_size: int = 120,
        mag_max: float = 4.5,
        show_detected_stars: bool = True,
        save_path: str = None,
    ):
        """
        Plot a zoomed section of the image with catalogue predictions and,
        optionally, detected stars overlaid.

        Parameters
        ----------
        xlim, ylim : tuple or None
            Explicit plot limits, e.g. xlim=(600, 900), ylim=(500, 800).
        centre : tuple or None
            Alternative to xlim/ylim. Gives zoom centre as (x, y).
        half_size : int
            Half-width of zoom box if using centre.
        mag_max : float
            Only plot catalogue stars brighter than this magnitude.
        show_detected_stars : bool
            If True, overlay all detected stars in the zoom window.
        """
        if self.image_data is None or self.obstime is None or self.location is None:
            self.open()

        img = np.asarray(self.image_data, dtype=np.float32)

        sky_xc = 791.11
        sky_yc = 621.99
        sky_r = 530.89
        mask_params = [sky_xc, sky_yc, sky_r]

        stars = self.find_stars(
            img=img,
            mask_params=mask_params,
            cfg=cfg,
            plot=False,
        )
        if len(stars) == 0:
            raise RuntimeError("No stars detected.")

        src_xy = np.vstack([stars["x"], stars["y"]]).T.astype(float)

        catalog, mask, alt_all, az_all = self.catalog_altaz()
        cat = catalog[mask]
        alt = alt_all[mask].astype(float)
        az = az_all[mask].astype(float)
        vt = cat["vt_mag"].astype(float)

        keep = (alt >= alt_min_deg) & np.isfinite(vt) & (vt <= mag_max)
        cat = cat[keep]
        alt = alt[keep]
        az = az[keep]
        vt = vt[keep]

        if len(cat) == 0:
            raise RuntimeError(
                f"No catalogue stars remain for alt_min_deg={alt_min_deg:.1f} deg "
                f"and mag_max={mag_max:.1f}."
            )

        u_pred, v_pred = self.model_altaz_to_uv(alt, az, self.model)
        uv_pred = np.vstack([u_pred, v_pred]).T.astype(float)

        H, W = img.shape

        if xlim is None or ylim is None:
            if centre is None:
                cx, cy = self.model.xc, self.model.yc
            else:
                cx, cy = centre

            xlim = (max(0, cx - half_size), min(W, cx + half_size))
            ylim = (max(0, cy - half_size), min(H, cy + half_size))

        x1, x2 = xlim
        y1, y2 = ylim

        finite = np.isfinite(img)
        if np.any(finite):
            vmin, vmax = np.percentile(img[finite], [1, 99.7])
        else:
            vmin, vmax = 0, 1

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

        in_win_cat = (
            (uv_pred[:, 0] >= x1)
            & (uv_pred[:, 0] <= x2)
            & (uv_pred[:, 1] >= y1)
            & (uv_pred[:, 1] <= y2)
        )
        ax.scatter(
            uv_pred[in_win_cat, 0],
            uv_pred[in_win_cat, 1],
            s=45,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.2,
            label=fr"Catalogue predictions ($m_V \leq {mag_max:.1f}$)",
        )

        if show_detected_stars:
            in_win_src = (
                (src_xy[:, 0] >= x1)
                & (src_xy[:, 0] <= x2)
                & (src_xy[:, 1] >= y1)
                & (src_xy[:, 1] <= y2)
            )
            ax.scatter(
                src_xy[in_win_src, 0],
                src_xy[in_win_src, 1],
                s=18,
                marker="o",
                linewidths=0.6,
                facecolors="none",
                edgecolors="red",
                label="Detected stars",
            )

        ax.set_xlim(x1, x2)
        ax.set_ylim(y1, y2)
        ax.set_xlabel("x [pix]", size=12)
        ax.set_ylabel("y [pix]", size=12)
        ax.legend(loc="best")

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    d = CalibrationDiagnostics(
        file_path="/Users/samcavens/Documents/4th/SHP/fits_playground/Raw_files/image-20231209-211719.fits",
        output_dir="/Users/samcavens/Documents/4th/SHP/fits_playground",
        cat_path="/Users/samcavens/Downloads/asu.fit",
    )

    result = d.plot_residual_vs_radius(
        cfg=ExtractionConfig(),
        alt_min_deg=20.0,
        bin_width_px=25,
        min_per_bin=5,
        save_path="/Users/samcavens/Documents/4th/SHP/fits_playground/residual_vs_radius.png",
    )

    print(result["overall_rmse_px"])
    print(result["n_matches"])

    d.plot_catalog_overlay_zoom(
        alt_min_deg=20.0,
        centre=(771.11, 622),
        half_size=350,
        mag_max=4.5,
        save_path="/Users/samcavens/Documents/4th/SHP/fits_playground/catalog_overlay_zoom_mag45.png",
    )
