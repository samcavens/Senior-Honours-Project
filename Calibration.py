import os
import json

import numpy as np
import matplotlib.pyplot as plt
import sep
from pathlib import Path

from astropy.io import fits
from astropy import units as u
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.table import Table
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from dataclasses import asdict
from types import SimpleNamespace
import numpy.ma as ma




class ImageParams:
    def __init__(self, xc, yc, s, theta_north_deg,  k, dr, dt):
        self.xc= xc                 # zenith pixel x
        self.yc=yc                 # zenith pixel y

    
        self.s=s                 # zenith-to-horizon radius in pixels, calculate for (xc,yc) and r 
        self.theta_north_deg=theta_north_deg    # north offset (deg)
        
        # projection coefficients (odd polynomial)
        self.k=k           # list of length 5: k0..k4

        # radial distortion coefficients (7)
        self.dr= dr       # δr1..δr7

        # tangential distortion coefficients (7)
        self.dt= dt        # δθ1..δθ7

@dataclass
class ExtractionConfig:
    # SEP background settings
    bw: int = 64
    bh: int = 64
    fw: int = 5
    fh: int = 5

    # SEP detection settings
    thresh_sigma: float =  1.6  # lower -> more detections (can raise false positives), 1.7 for balancer
    minarea: int = 8 #15 for balance
    maxarea: int= 200

    # parameters to tune

    rsn_min: float = 3.0 #rsn>rsn_min 3 for balance
    tmetric_max: float = 6.2 #tmetric < tmetric_max 15 for balance
    
    rsn_bright: float = 10.0 # "if it's this significant, keep it even if big"
    a_max: float =10.0
    # Matching
    match_radius_px: float = 15.0

    # Qvis options
    nside: int = 8
    alt_min_deg: float = 25.0   # ignore below this altitude


class Calibrate: 
    def __init__(self, file_path: str, solved_file: str, output_dir: str, cat_path:str): 
        self.file_path=file_path 
        self.solved_file=solved_file
        self.output_dir = output_dir

        self.cat_path=cat_path

        self.base, self.ext = os.path.splitext(self.file_path)

        self.header = None
        self.image_data= None

        # Populated by read_metadata()
        self.obstime = None
        self.location = None
        self.elevation = None 
        
        self.zenith_RA=None
        self.zenith_Dec=None

        # Populated by ra_dec_guess()
        self.ra_deg = None
        self.dec_deg = None

        
    def open(self): 
        with fits.open(self.file_path) as hdulist:
             primary_header = hdulist[0].header 
             self.image_data = hdulist[0].data


        date_time = primary_header.get("DATE-OBS") #
        long = primary_header.get("LONGITUD")
        lat  = primary_header.get("LATITUDE")
        height_m  = primary_header.get("ALTITUDE", 0)*1000

        t     = Time(date_time, format="isot", scale="utc")   
        loc   = EarthLocation(lat=lat*u.deg, lon=long*u.deg, height=height_m*u.m) 
        elevation = float(height_m)

        self.obstime= t
        self.location= loc
        self.header=primary_header

        self.elevation=elevation #assume in km
        

        return t, loc, elevation
    
    def load_wcs_and_offsets(self):
        with fits.open(self.solved_file) as hdul:
            hdr = hdul[0].header
            w = WCS(hdr)
            data = hdul[0].data

        x0 = int(hdr.get("HIERARCH CUT_X0", 0))
        y0 = int(hdr.get("HIERARCH CUT_Y0", 0))
        return w, (x0, y0), data, hdr
    
    def wcs_guess(self):
            zenith_altaz = SkyCoord( alt=90*u.deg, az=0*u.deg, frame=AltAz(obstime=self.obstime, location=self.location))
            zenith_icrs  = zenith_altaz.icrs

            zenith_ra_deg  = zenith_icrs.ra.deg
            zenith_dec_deg = zenith_icrs.dec.deg

            print(zenith_ra_deg, zenith_dec_deg)

            return zenith_ra_deg, zenith_dec_deg
    
    @staticmethod
    def circular_mask( shape, xc, yc, r, margin=0):
        """
        Return mask for SEP:
        True  = ignore pixel
        False = use pixel
        """
        yy, xx = np.indices(shape)
        rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)

        return rr > (r - margin)   # True OUTSIDE circle
        
    def find_stars(self, img,  mask_params: np.array, cfg: ExtractionConfig = ExtractionConfig(), return_bkg: bool = False,
                plot: bool = False, fov_margin_px: int = 10):

        img = np.asarray(img, dtype=np.float32) #load in img a
        sky_xc, sky_yc, sky_r= mask_params #unpack mask params

        mask = self.circular_mask(img.shape, sky_xc, sky_yc, sky_r, margin=fov_margin_px)

        #use sep to get background, mask in sep
        bkg = sep.Background(img, mask=mask, bw=cfg.bw, bh=cfg.bh, fw=cfg.fw, fh=cfg.fh) # measure a spatially varying background on the image
        bkg_img = bkg.back() #evaluate background as 2d array, same size as image
        rms_img = bkg.rms() #noise across sky, use for detection threshold

        data_sub = img - bkg_img
        thresh = cfg.thresh_sigma * bkg.globalrms

        valid = ~mask
        above = (data_sub > thresh) & valid
        frac = above.sum() / valid.sum()
        
        # if more than e.g. 50% of *unmasked* pixels are above thresh, likely whole image is noise
        print(f"frac above bkg threshold {frac}")
        if frac > 0.50:
            # treat as bad/empty frame 
            stars = Table(names=["x","y","flux","a","b","theta","npix"], dtype=[float]*7)
            print(" more than e.g. 50% of *unmasked* pixels are above thresh, likely whole image is noise")
            return (stars, bkg_img, rms_img) if return_bkg else stars

        try:
            sources = sep.extract(data_sub, thresh=thresh, minarea=cfg.minarea, mask=mask, 
            deblend_nthresh=32,    # Required to trigger deblending
            deblend_cont=0.0005,   # Note: SEP uses 'deblend_cont', not 'DEBLEND_MINCONT'
            clean=True,            # Removes artifacts near the edges
            clean_param=1.0)
                
        except Exception as e:
            if "internal pixel buffer full" in str(e):
                # SEP overwhelmed, treat as no usable detections
                stars = Table(
                    names=["x","y","flux","a","b","theta","npix"],
                    dtype=[float]*7
                )
                print("internal pixel buffer full")
                return (stars, bkg_img, rms_img) if return_bkg else stars
            else:
                raise
    
        # peak, local bkg and local rms
        x0 = np.clip(np.rint(sources["x"]).astype(int), 0, img.shape[1] - 1)
        y0 = np.clip(np.rint(sources["y"]).astype(int), 0, img.shape[0] - 1)

        bkg_at = bkg_img[y0, x0]
        rms_at = rms_img[y0, x0]
        rms_at = np.where(rms_at > 0, rms_at, np.nan)

        # --- local-maximum peak in a small window around centroid ---
        half = 2  # 2 -> 5x5 window. try 1 (3x3) if you prefer
        H, W = img.shape
        peak = np.empty(len(x0), dtype=float)

        for i, (xi, yi) in enumerate(zip(x0, y0)):
            x1 = max(0, xi - half); x2 = min(W, xi + half + 1)
            y1 = max(0, yi - half); y2 = min(H, yi + half + 1)
            peak[i] = np.nanmax(img[y1:y2, x1:x2])

        rsn = (peak - bkg_at) / rms_at

        npix = sources["npix"].astype(float)

        # avoid divide-by-zero / negative rsn blowing things up
        tmetric = np.full_like(rsn, np.inf, dtype=float)
        good_rsn = np.isfinite(rsn) & (rsn > 0)
        tmetric[good_rsn] = npix[good_rsn] / rsn[good_rsn]

        # parameters to tune
        rsn_min, tmetric_max, rsn_bright = cfg.rsn_min, cfg.tmetric_max, cfg.rsn_bright

        # include shape/area cuts
        a = sources["a"]  # SEP's semi-major axis (pixels)

        keep = (np.isfinite(rsn) & (rsn > rsn_min) & np.isfinite(a) &
            (
                (((npix <= cfg.maxarea) & (a <= cfg.a_max) & (tmetric < tmetric_max)))
                | (rsn > rsn_bright)
            )
        )

        sources = sources[keep]
        if len(sources) == 0: #finish function if nothing detected (stops errors)
            stars = Table(names=["x","y","flux","a","b","theta","npix"], dtype=[float]*7)
            print(stars)
            return (stars, bkg_img, rms_img) if return_bkg else stars

        # Pack into a nice table
        stars = Table()
        stars["x"] = sources["x"]
        stars["y"] = sources["y"]
        stars["flux"] = sources["flux"]
        stars["a"] = sources["a"]
        stars["b"] = sources["b"]
        stars["theta"] = sources["theta"]
        stars["npix"] = sources["npix"]

        actual_image_centre= np.array(np.shape(self.image_data))//2 #y,x
        measured_image_centre=(621.99,791.11) #y,x
        xc, yc=758.19229, 631.70484 #zenith xc,yc

        dx = xc - measured_image_centre[1]
        dy = yc - measured_image_centre[0]

        offset = np.sqrt(dx**2 + dy**2)

        s = sky_r - offset

        if plot:
            # Robust display stretch (similar vibe to DS9 zscale)
            img_show = img.copy()
            finite = np.isfinite(img_show)
            if np.any(finite):
                vmin, vmax = np.percentile(img_show[finite], [1, 99.7])
            else:
                vmin, vmax = 0, 1

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img_show, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
            ax.scatter(stars["x"], stars["y"], s=12, facecolors="none", edgecolors="lime", linewidths=0.8)
            ax.scatter(xc, yc,
               s=25,
               marker="x",
               c="red",
               linewidths=0.5,
               label="Zenith (xc, yc)")
            ax.set_title(f"SEP detections after screening: N={len(stars)}  (thresh={cfg.thresh_sigma}σ)")
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("y [pix]")


            plt.tight_layout()
            plt.show()



        return (stars, bkg_img, rms_img) if return_bkg else stars

    
    def load_catalog(self):
        """
        Load Johnson+1966 UBV catalog from either VizieR FITS table (asu.fit / .fits)  -> columns like _RA, _DE, Vmag
        Outputt: id, ra_deg, dec_deg, vt_mag  (we use Vmag as vt_mag proxy)
        """
        path=self.cat_path
        table = Table.read(path)

        #   LID, Vmag, _RA, _DE, Lid
        ra_col = "_RA" 
        dec_col = "_DE" 
        v_col = "Vmag"       

        ra_deg = np.array(table[ra_col], dtype=float)
        dec_deg = np.array(table[dec_col], dtype=float)
        vt_mag = np.array(table[v_col], dtype=float)
        ids = np.array(table["LID"]).astype(str)


        rec = np.core.records.fromarrays(
            [ids, ra_deg, dec_deg, vt_mag],
            names=["id", "ra_deg", "dec_deg", "vt_mag"],
        )

        return rec

    def catalog_altaz(self):
        """Compute alt/az for catalog stars at time/location. Returns (mask_above_horizon, alt_deg, az_deg)."""
        catalog=self.load_catalog()
        obstime=self.obstime
        location=self.location
        sc = SkyCoord(ra=catalog["ra_deg"] * u.deg, dec=catalog["dec_deg"] * u.deg, frame="icrs")
        altaz = sc.transform_to(AltAz(obstime=obstime, location=location))
        alt = altaz.alt.deg
        az = altaz.az.deg
        mask = alt > 0.0
        return catalog, mask, alt, az    

    def match_using_wcs(self, cfg: ExtractionConfig, plot: bool = False):
        # 1) load solved crop: WCS + offsets + crop image
        w, (x0, y0), crop_img, hdr = self.load_wcs_and_offsets()

        # 2) circle params in CROP coords
        cx_cut = float(hdr["HIERARCH CIRC_CX"]) - float(hdr["HIERARCH CUT_X0"])
        cy_cut = float(hdr["HIERARCH CIRC_CY"]) - float(hdr["HIERARCH CUT_Y0"])
        r      = float(hdr["HIERARCH CIRC_R"])
        mask_params_crop = [cx_cut, cy_cut, r]

        # 3) detect stars in crop coords (uses same circle mask)
        stars_crop = self.find_stars(img=crop_img, mask_params=mask_params_crop, cfg=cfg, plot=plot)
        if len(stars_crop) == 0:
            return [], self.load_catalog()[0:0]

        src_xy_crop = np.vstack([stars_crop["x"], stars_crop["y"]]).T.astype(float)

        # 4) load catalog
        catalog = self.load_catalog()

        # 5) catalog RA/Dec -> crop pixels using WCS
        sc = SkyCoord(ra=catalog["ra_deg"] * u.deg, dec=catalog["dec_deg"] * u.deg, frame="icrs")
        x_pred, y_pred = w.world_to_pixel(sc)  # 0-based crop pixel coords
        cat_xy_crop = np.vstack([x_pred, y_pred]).T.astype(float)

        # 6) keep only catalog predictions inside the circle (and finite)
        finite = np.isfinite(cat_xy_crop[:, 0]) & np.isfinite(cat_xy_crop[:, 1])
        dx = cat_xy_crop[:, 0] - cx_cut
        dy = cat_xy_crop[:, 1] - cy_cut
        inside_circle = (dx * dx + dy * dy) <= (r * r)

        keep = finite & inside_circle
        cat2 = catalog[keep]
        cat_xy_crop2 = cat_xy_crop[keep]
        vt2 = cat2["vt_mag"].astype(float)

        # 7) match in crop coords
        recognized_crop = self.match_mnn(
            cat_xy_crop2, vt2, cat2["id"], src_xy_crop, match_radius_px=cfg.match_radius_px
        )

        # 8) convert matched observed pixels to FULL coords
        recognized_full = [(cid, vt, x + x0, y + y0) for (cid, vt, x, y) in recognized_crop]

        
        # ------------------------
        # PLOTS 
        # ------------------------
        if plot:
            # --- Stretch for display ---
            img = np.asarray(crop_img, dtype=np.float32)
            finite_img = np.isfinite(img)
            if np.any(finite_img):
                vmin, vmax = np.percentile(img[finite_img], [1, 99.7])
            else:
                vmin, vmax = 0, 1

            # Build arrays for matched points in crop coords
            if len(recognized_crop) > 0:
                matched_src = np.array([[x, y] for (_, _, x, y) in recognized_crop], dtype=float)
                matched_ids = [str(cid) for (cid, _, _, _) in recognized_crop]
                id_to_i = {str(cid): i for i, cid in enumerate(cat2["id"])}
                matched_cat = np.array([cat_xy_crop2[id_to_i[cid]] for cid in matched_ids if cid in id_to_i], dtype=float)
            else:
                matched_src = np.zeros((0, 2), dtype=float)
                matched_cat = np.zeros((0, 2), dtype=float)

            # --- Plot 1: crop overlay ---
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            # Circle boundary
            ax.add_patch(plt.Circle((cx_cut, cy_cut), r, edgecolor="white", facecolor="none", linewidth=1.2, alpha=0.8))

            # All SEP detections
            ax.scatter(src_xy_crop[:, 0], src_xy_crop[:, 1],
                    s=12, facecolors="none", edgecolors="lime", linewidths=0.7, label="SEP detections (crop)")

            # All catalog predictions inside circle
            ax.scatter(cat_xy_crop2[:, 0], cat_xy_crop2[:, 1],
                    s=8, c="cyan", alpha=0.6, label="Catalog preds (inside circle)")

            # Matched detections + matched catalog preds
            if len(matched_src) > 0:
                ax.scatter(matched_src[:, 0], matched_src[:, 1],
                        s=40, facecolors="none", edgecolors="red", linewidths=1.2, label="Matched detections")
            if len(matched_cat) > 0:
                ax.scatter(matched_cat[:, 0], matched_cat[:, 1],
                        s=35, marker="x", c="yellow", linewidths=1.5, label="Matched catalog preds")

            ax.set_title(f"WCS match on crop: {len(recognized_crop)} matches | CUT offset=({x0},{y0})")
            ax.set_xlabel("x (crop pix)")
            ax.set_ylabel("y (crop pix)")
            ax.legend(loc="lower left")
            plt.tight_layout()
            plt.show()

            # --- Plot 2: full-image check (if full image loaded) ---
            if self.image_data is not None and len(recognized_full) > 0:
                full = np.asarray(self.image_data, dtype=np.float32)
                finite_full = np.isfinite(full)
                if np.any(finite_full):
                    vmin2, vmax2 = np.percentile(full[finite_full], [1, 99.7])
                else:
                    vmin2, vmax2 = 0, 1

                full_xy = np.array([[x, y] for (_, _, x, y) in recognized_full], dtype=float)

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(full, origin="lower", cmap="gray", vmin=vmin2, vmax=vmax2)
                ax.scatter(full_xy[:, 0], full_xy[:, 1],
                        s=40, facecolors="none", edgecolors="red", linewidths=1.2,
                        label="Matched detections (full coords)")
                ax.set_title("Matched detections mapped back to FULL image coords")
                ax.set_xlabel("x (full pix)")
                ax.set_ylabel("y (full pix)")
                ax.legend(loc="lower left")
                plt.tight_layout()
                plt.show()
        return recognized_full, cat2 

    # 
    # Camera transform 
    #
    def model_altaz_to_uv(self, alt_deg, az_deg, model):
        def poly_rho_to_r(rho, k):
            r = np.zeros_like(rho, dtype=float)
            powers = [1, 3, 5, 7, 9]
            for ki, p in zip(k, powers):
                r += ki * np.power(rho, p)
            return r

        def delta_terms(rho_n, theta, d):
            # rho_n should be dimensionless in [0,1]
            d1, d2, d3, d4, d5, d6, d7 = d
            A = d1 * rho_n + d2 * rho_n**3 + d3 * rho_n**5
            B = (d4 * np.cos(theta) + d5 * np.sin(theta) +
                d6 * np.cos(2 * theta) + d7 * np.sin(2 * theta))
            return A * B

        alt = np.deg2rad(alt_deg) #al
        az = np.deg2rad(az_deg)

        rho = model.s * (1.0 - (2.0 * alt / np.pi))
        rho_n = rho / model.s #dimensionless radius for distortion series

        theta = az - np.deg2rad(model.theta_north_deg)

        k = np.asarray(model.k, dtype=float)
        r = poly_rho_to_r(rho, k)

        dr = np.asarray(model.dr, dtype=float)
        dt = np.asarray(model.dt, dtype=float)

        delta_r = delta_terms(rho_n, theta, dr)
        delta_t = delta_terms(rho_n, theta, dt)

        r_p = r + delta_r
        t_p = theta + delta_t

        x = r_p * np.cos(t_p)
        y = r_p * np.sin(t_p)

        u = x + model.xc
        v = y + model.yc
        return u, v
    
    @staticmethod
    def match_catalog_to_sources(cat_uv, cat_vt, cat_id, src_xy, match_radius_px):
        """
        'greedy' catalog star (brightest first) picks nearest detection
        
        :param cat_uv: (N,2) catalog star positions in pixels (u,v)
        :param cat_vt:(N,)  catalog magnitudes (brightness)cat_id,   
        :param cat_id:(N,)  catalog identifiers
        :param src_xy: (M,2) detected source positions (x,y)
        :param cfg:  for matching tolerance in pixels
        """

        if len(cat_uv) == 0 or len(src_xy) == 0:
            print("Len of cat_uv or src_xy is zero")
            return []

        order = np.argsort(cat_vt)  # brighter first
        src = src_xy.copy()

        recognized: List[Tuple[str, float, float, float]] = []
        used = np.zeros(len(src), dtype=bool)

        for idx in order:
            cu, cv = cat_uv[idx]
            du = src[:, 0] - cu
            dv = src[:, 1] - cv
            dist2 = du * du + dv * dv
            dist2[used] = np.inf

            j = int(np.argmin(dist2))
            d = np.sqrt(dist2[j])
            if np.isfinite(d) and d < match_radius_px:
                used[j] = True
                recognized.append((str(cat_id[idx]), float(cat_vt[idx]), float(src[j, 0]), float(src[j, 1])))

        return recognized
    
    @staticmethod
    def match_mnn(cat_uv, cat_vt, cat_id, src_xy, match_radius_px=6.0):
        """
        Mutual nearest-neighbour matching.
        Returns list of (id, vt, x_obs, y_obs).
        """
        if len(cat_uv) == 0 or len(src_xy) == 0:
            return []

        cat_uv = np.asarray(cat_uv, float)
        src_xy = np.asarray(src_xy, float)

        tree_src = cKDTree(src_xy)
        tree_cat = cKDTree(cat_uv)

        # catalog -> nearest source
        d_cs, j_src = tree_src.query(cat_uv, k=1)   # for each catalog i, nearest src index
        # source -> nearest catalog
        d_sc, i_cat = tree_cat.query(src_xy, k=1)   # for each source j, nearest cat index

        # mutual condition: source chosen by catalog must also choose that catalog
        cat_idx = np.arange(len(cat_uv))
        mutual = (i_cat[j_src] == cat_idx) & (d_cs < match_radius_px)

        # candidates
        cand_i = np.where(mutual)[0]
        if cand_i.size == 0:
            return []

        # If multiple cats would still collide (rare with mutual, but possible),
        # keep the closest one-to-one by sorting by distance
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
    def _pack_model(model: ImageParams) -> np.ndarray:
        return np.array(
            [model.s, model.theta_north_deg, model.xc, model.yc] +
            list(model.k) + list(model.dr) + list(model.dt),
            dtype=float
        )
    
    @staticmethod
    def _unpack_model(x: np.ndarray) -> ImageParams:
        x = np.asarray(x, dtype=float)
        s, theta_n, xc, yc = x[0:4]
        k = x[4:9].tolist()
        dr = x[9:16].tolist()
        dt = x[16:23].tolist()
        return ImageParams(xc, yc, s, theta_n, k, dr,dt
                           )
    
    def print_model(self, model, label="model"):
        def wrap(theta):
            return ((theta + 180.0) % 360.0) - 180.0
        
        print(f"\n--- {label} ---")
        print(f"xc = {model.xc:.3f}, yc = {model.yc:.3f}")
        print(f"s  = {model.s:.3f}")
        print(f"theta_north_deg = {model.theta_north_deg:.3f} (wrapped {wrap(model.theta_north_deg):.3f})")
        print("k  =", [float(f"{v:.6g}") for v in model.k])
        print("dr =", [float(f"{v:.6g}") for v in model.dr])
        print("dt =", [float(f"{v:.6g}") for v in model.dt])

    def save_model(self, model, path):
        payload = {
            "xc": float(model.xc),
            "yc": float(model.yc),
            "s": float(model.s),
            "theta_north_deg": float(model.theta_north_deg),
            "k": [float(v) for v in model.k],
            "dr": [float(v) for v in model.dr],  # will be zeros
            "dt": [float(v) for v in model.dt],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2) 

    def _residuals_uv(self, x, alt_deg, az_deg, uv_obs):
        """Flattened residuals [du0,dv0, du1,dv1, ...]."""
        model = self._unpack_model(x)
        u_pred, v_pred = self.model_altaz_to_uv(alt_deg, az_deg, model)
        uv_pred = np.vstack([u_pred, v_pred]).T
        return (uv_pred - uv_obs).ravel()
    

    def _fit(self, x0, mask_free, alt_deg, az_deg, uv_obs, robust=True):
        """Fit only parameters where mask_free==True."""
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
    
    def calibrate_using_wcs(self,
        init_model: ImageParams,
        cfg: ExtractionConfig = ExtractionConfig(),
        min_matches: int = 15,
        staged: bool = True,
        plot: bool = False,
    ):
        """
        End-to-end calibration from one clear image.
        Returns (best_model, diagnostics_dict).
        """
        # ensure full image + metadata loaded (time/location come from raw full FITS)
        if self.image_data is None or self.obstime is None or self.location is None:
            self.open()

        # get high-confidence matches using WCS on the solved crop
        recognized_full, cat2 = self.match_using_wcs(cfg=cfg, plot=plot)

        if len(recognized_full) < min_matches:
            raise RuntimeError(f"Too few WCS matches ({len(recognized_full)}). Try match_radius_px=5–10 or lower thresh_sigma.")

        #  map catalog IDs -> rows in cat2
        id_to_idx = {str(cid): i for i, cid in enumerate(cat2["id"])}

        #  build arrays for fitting: (alt,az) and observed full-frame pixels
        alt_m, az_m, uv_obs = [], [], []
        for cid, vt_mag, xobs, yobs in recognized_full:
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
        az_m  = np.asarray(az_m, dtype=float)
        uv_obs = np.asarray(uv_obs, dtype=float)

        if len(alt_m) < min_matches:
            raise RuntimeError(f"After ID mapping, only {len(alt_m)} pairs remain. (ID mismatch?)")

        # run the same least-squares fit as before
        x = self._pack_model(init_model)

        if staged:
            mask_k = np.zeros_like(x, bool); mask_k[4:9] = True
            x, _ = self._fit(x, mask_k, alt_m, az_m, uv_obs)

            mask_d = np.zeros_like(x, bool); mask_d[9:23] = True
            x, _ = self._fit(x, mask_d, alt_m, az_m, uv_obs)

            mask_g = np.zeros_like(x, bool); mask_g[0:4] = True
            x, _ = self._fit(x, mask_g, alt_m, az_m, uv_obs)

            mask_all = np.ones_like(x, bool)
            x, final_res = self._fit(x, mask_all, alt_m, az_m, uv_obs)
        else:
            mask_all = np.ones_like(x, bool)
            x, final_res = self._fit(x, mask_all, alt_m, az_m, uv_obs)

        best_model = self._unpack_model(x)

        # diagnostics and  overlay plot on full image
        rms_px = float(np.sqrt(np.mean(final_res.fun**2)))
        diag = {
            "n_wcs_matches": int(len(recognized_full)),
            "n_used_in_fit": int(len(alt_m)),
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

            ax.scatter(uv_obs[:, 0], uv_obs[:, 1],
                    s=35, facecolors="none", edgecolors="lime", linewidths=1.0,
                    label="Obs (WCS-matched, full coords)")
            ax.scatter(u_fit, v_fit,
                    s=18, c="yellow", label="Pred (camera model)")

            ax.set_title(f"WCS-seeded calibration (RMS={rms_px:.2f}px, N={len(alt_m)})")
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("y [pix]")
            ax.legend(loc="lower left")
            plt.tight_layout()
            plt.show()

        return best_model, diag
        
    def calibrate(
        self,
        init_model: ImageParams,
        cfg: ExtractionConfig = ExtractionConfig(),
        min_matches: int = 15,
        staged: bool = True,
        plot: bool = False,
    ):
        """
        End-to-end calibration from one clear image.
        Returns (best_model, diagnostics_dict).
        """

        # ensure metadata + image loaded 
        if self.image_data is None or self.obstime is None or self.location is None:
            self.open()

        #  detect stars 
        sky_xc=791.11
        sky_yc=621.99
        sky_r=530.89  
        mask_params=[sky_xc, sky_yc, sky_r]  
        stars = self.find_stars(img=self.image_data, mask_params=mask_params, cfg=cfg, plot=plot)
        if len(stars) == 0:
            raise RuntimeError("No stars detected after screening.")
        src_xy = np.vstack([stars["x"], stars["y"]]).T.astype(float)

        #  catalog -> alt/az 
        catalog, mask, alt_all, az_all = self.catalog_altaz()
        cat = catalog[mask]
        alt = alt_all[mask].astype(float)
        az  = az_all[mask].astype(float)
        vt  = cat["vt_mag"].astype(float)

        #  initial projection 
        u0, v0 = self.model_altaz_to_uv(alt, az, init_model)
        cat_uv0 = np.vstack([u0, v0]).T

        #  match catalog to sources
        #recognized = self.match_catalog_to_sources(cat_uv=cat_uv0, cat_vt=vt, cat_id=cat["id"], src_xy=src_xy, cfg=cfg)
        recognized = self.match_mnn(cat_uv0, vt, cat["id"], src_xy, match_radius_px=cfg.match_radius_px)
        if len(recognized) < min_matches:
            raise RuntimeError(
                f"Too few matches ({len(recognized)}). "
                f"Try looser detection/match params (thresh_sigma down, match_radius_px up)."
            )

        # Build matched arrays: alt_m, az_m, uv_obs
        id_to_idx = {str(cid): i for i, cid in enumerate(cat["id"])}
        alt_m, az_m, uv_obs = [], [], []
        for cid, vt_mag, uobs, vobs in recognized:
            i = id_to_idx.get(str(cid))
            if i is None:
                continue
            alt_m.append(alt[i])
            az_m.append(az[i])
            uv_obs.append([uobs, vobs])

        alt_m = np.asarray(alt_m, float)
        az_m  = np.asarray(az_m, float)
        uv_obs = np.asarray(uv_obs, float)

        # least-squares fit 
        x = self._pack_model(init_model)

        # parameter indices in packed vector:
        # 0:s, 1:theta, 2:xc, 3:yc, 4:9=k(5), 9:16=dr(7), 16:23=dt(7)
        if staged:
            # Fit k only
            mask_k = np.zeros_like(x, bool); mask_k[4:9] = True
            x, res_k = self._fit(x, mask_k, alt_m, az_m, uv_obs)

            # Fit dr/dt only
            mask_d = np.zeros_like(x, bool); mask_d[9:23] = True
            x, res_d = self._fit(x, mask_d, alt_m, az_m, uv_obs)

            # Fit globals s/theta/xc/yc
            mask_g = np.zeros_like(x, bool); mask_g[0:4] = True
            x, res_g = self._fit(x, mask_g, alt_m, az_m, uv_obs)

            # Fit all
            mask_all = np.ones_like(x, bool)
            x, res_all = self._fit(x, mask_all, alt_m, az_m, uv_obs)

            final_res = res_all
        else:
            mask_all = np.ones_like(x, bool)
            x, final_res = self._fit(x, mask_all, alt_m, az_m, uv_obs)

        best_model = self._unpack_model(x)

        # --- 6) diagnostics ---
        rms_px = float(np.sqrt(np.mean(final_res.fun**2)))
        diag = {
            "n_detected": int(len(stars)),
            "n_matched": int(len(alt_m)),
            "rms_px": rms_px,
            "success": bool(final_res.success),
            "message": str(final_res.message),
        }

        # Optional plot: predicted vs observed for matched stars
        if plot:
            # show the original image under the points
            img = np.asarray(self.image_data, dtype=np.float32)
            finite = np.isfinite(img)
            if np.any(finite):
                vmin, vmax = np.percentile(img[finite], [1, 99.7])
            else:
                vmin, vmax = 0, 1

            u_fit, v_fit = self.model_altaz_to_uv(alt_m, az_m, best_model)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            # observed matched detections
            ax.scatter(uv_obs[:, 0], uv_obs[:, 1],
                    s=30, facecolors="none", edgecolors="lime",
                    linewidths=1.0, label="Observed (matched detections)")

            # predicted positions
            ax.scatter(u_fit, v_fit,
                    s=18, c="yellow", label="Predicted (model)")

            ax.set_title(f"Overlay: matched stars (RMS={rms_px:.2f}px)")
            ax.set_xlabel("x [pix]")
            ax.set_ylabel("y [pix]")
            ax.legend(loc="lower left")
            plt.tight_layout()
            plt.show()

            best_model = self._unpack_model(x)
            self.print_model(best_model, label="best_model")
            self.save_model(best_model, "/Users/samcavens/Documents/4th/SHP/fits_playground/camera_calibration2.json")

        return best_model, diag

    def calibrate_two_stage(
        self,
        init_model: ImageParams,
        cfg_wcs: ExtractionConfig,
        cfg_full: ExtractionConfig,
        plot: bool = False,
    ):
        # Stage A: WCS-seeded
        model_seed, diag_seed = self.calibrate_using_wcs(
            init_model=init_model,
            cfg=cfg_wcs,
            staged=True,
            plot=False
        )

        # Stage B: full-image refinement (your original calibrate, but start from model_seed)
        model_full, diag_full = self.calibrate(
            init_model=model_seed,
            cfg=cfg_full,
            staged=True,
            plot=plot
        )

        return model_full, {"seed": diag_seed, "full": diag_full}
    
    

    def plot_altitude_map(
        self,
        sky_mask_params=(791.11, 621.99, 530.89),   # (sky_xc, sky_yc, sky_r)
        step: int = 4,                              # downsample factor for speed
        plot_on_image: bool = False,
    ):
        """
        Plot altitude distribution in image coordinates:
        pixel color = altitude (deg). Concentric circles => good mapping.

        Uses radial inverse of the fitted rho->r polynomial.
        """

        model_path = Path("/Users/samcavens/Documents/4th/SHP/fits_playground/camera_calibration.json")

        # load calibration model
        with open(model_path, "r") as f:
            calib = json.load(f)

        # turn dict into object with attributes: model.xc, model.yc, etc.
        model = SimpleNamespace(**calib)

        if self.image_data is None:
            self.open()

        img = np.asarray(self.image_data, dtype=np.float32)
        H, W = img.shape

        sky_xc, sky_yc, sky_r = map(float, sky_mask_params)

        # grid of pixel coords (downsampled)
        yy, xx = np.mgrid[0:H:step, 0:W:step]
        xx = xx.astype(float)
        yy = yy.astype(float)

        # distance from fitted zenith
        r_pix = np.sqrt((xx - model.xc) ** 2 + (yy - model.yc) ** 2)

        # mask outside physical sky circle
        inside = ((xx - sky_xc) ** 2 + (yy - sky_yc) ** 2) <= (sky_r ** 2)

        # forward polynomial: r(rho) = k1*rho + k3*rho^3 + ...
        k = np.asarray(model.k, dtype=float)

        def r_from_rho(rho):
            powers = np.array([1, 3, 5, 7, 9], dtype=int)
            out = np.zeros_like(rho, dtype=float)
            for ki, p in zip(k, powers):
                out += ki * rho**p
            return out

        # invert r(rho) numerically via binary search
        rho_lo = np.zeros_like(r_pix, dtype=float)
        rho_hi = np.full_like(r_pix, float(model.s), dtype=float)

        for _ in range(30):
            rho_mid = 0.5 * (rho_lo + rho_hi)
            r_mid = r_from_rho(rho_mid)
            go_hi = r_mid > r_pix
            rho_hi[go_hi] = rho_mid[go_hi]
            rho_lo[~go_hi] = rho_mid[~go_hi]

        rho_est = 0.5 * (rho_lo + rho_hi)

        # convert rho -> altitude
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
        ax.add_patch(
            plt.Circle(
                (sky_xc, sky_yc),
                sky_r+1,
                edgecolor="white",
                facecolor="none",
                linewidth=3.0,
            )
        )

        ax.set_title("Altitude map in image coordinates")
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")
        cbar = plt.colorbar(im, ax=ax,shrink=0.6)
        cbar.set_label("Altitude [deg]")
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=300, bbox_inches="tight")

        plt.show()


"""

target_path = "/Users/samcavens/Documents/4th/SHP/fits_playground/Raw_files/image-20220930-234859.fits"
solved_path= "/Users/samcavens/Documents/4th/SHP/fits_playground/Solved_files/image-20220930-234859_crop_solved.fits"
output_path = "/Users/samcavens/Documents/4th/SHP/fits_playground"
catalog_path = "/Users/samcavens/Downloads/asu.fit"

solver = Calibrate(target_path, solved_path, output_path, catalog_path)
solver.open()

init = ImageParams(
    xc=758.19229, yc=631.70484,
    s=496.57, theta_north_deg=200.0,
    k=[1,0,1,0,0],
    dr = [1e-5, 0.0, 0.0,  1e-5, 0.0, 0.0, 0.0],
    dt = [1e-5, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0]    
)
cfg_wcs = ExtractionConfig(match_radius_px=5.0)
cfg_full=ExtractionConfig(match_radius_px=10.0)
#solver.match_using_wcs(cfg=ExtractionConfig, plot=True)
best_model, diag = solver.calibrate_two_stage(init_model=init,cfg_wcs=cfg_wcs, cfg_full=cfg_full, plot=True)
solver.plot_altitude_map(best_model)

print(diag)
print(best_model.xc, best_model.yc, best_model.s, best_model.theta_north_deg)

xc, yc=758.19229, 631.70484 #zenith xc,yc
s=496.5686692001941
#stars, bkg_img, rms_img = solver.find_stars(return_bkg=True, plot=True)
sky_xc=791.11
sky_yc=621.99
sky_r=530.89
"""
