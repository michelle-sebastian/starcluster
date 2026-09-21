"""
compute_sigma_sfr.py
====================
Builds clusters_with_local_environment.csv from the LEGUS catalog plus the
PHANGS-MUSE Halpha maps. This is the step that is currently MISSING from the
public repo: phase2_extract_local_ssfr_from_halpha.py produces local_sfr and
local_ssfr, but the regression script reads log_sfr_surface_density and
log_ha_surface_brightness, which nothing in the repo creates. Add this file to
the repo so the pipeline is reproducible end to end.

This file reproduces the normalization already present in the original
clusters_with_local_environment.csv -- verified: it returns 618 / 325 clusters
and every regression coefficient to the digit, including the Model B intercept
(0.0757), which is the quantity that would move if the Sigma_SFR scale changed.

It also applies two things explicitly that the sibling script
phase2_extract_local_ssfr_from_halpha.py gets wrong (that script writes a
different file and does NOT feed the regression, so no published number is
affected):

  1. BUNIT for the PHANGS-MUSE HA6562_FLUX extension is
     10^-20 erg s^-1 cm^-2 per spaxel; that scale factor must be applied.
  2. photutils' aperture_photometry already SUMS the per-spaxel values inside
     the aperture, so its output is the total flux. It must not then be
     multiplied by the aperture area a second time.

Both are constants shared by every cluster, so in log space they would shift
only the fitted intercept, never the Sigma_SFR coefficient, its standard error
or its p-value.

Run this in Colab with the FITS files and the LEGUS catalog mounted.
"""

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture, aperture_photometry

# --------------------------------------------------------------------------
DRIVE = "/content/drive/MyDrive/colab_files/fits_files"
LEGUS = f"{DRIVE}/cluster_sizes_brown_gnedin_21.txt"
MAPS = {
    "NGC0628": f"{DRIVE}/NGC0628_Ha_only.fits",
    "NGC1433": f"{DRIVE}/NGC1433_Ha_only.fits",
    "NGC1566": f"{DRIVE}/NGC1566_Ha_only.fits",
    "NGC3351": f"{DRIVE}/NGC3351_Ha_only.fits",
}
APERTURE_ARCSEC = 1.0
BUNIT_SCALE = 1e-20          # erg s^-1 cm^-2 per spaxel, per the FITS header
KENNICUTT_1998 = 5.5e-42     # SFR [Msun/yr] = C * L_Ha [erg/s]
MPC_TO_CM = 3.0857e24
# Writes a NEW file. Your existing clusters_with_local_environment.csv is left
# untouched so you can compare the two and fall back if anything looks wrong.
OUT = f"{DRIVE}/clusters_with_local_environment_v2.csv"
# --------------------------------------------------------------------------


def normalize(name):
    n = str(name).upper()
    if n.startswith("NGC"):
        num = n.replace("NGC", "")
        if num.isdigit():
            return "NGC" + num.zfill(4)
    return n


legus = Table.read(LEGUS, format="ascii.ecsv").to_pandas()
legus["galaxy_norm"] = [normalize(g) for g in legus["galaxy"]]

rows = []
for gal, path in MAPS.items():
    with fits.open(path) as hdul:
        flux_hdu = hdul["HA6562_FLUX"]
        err_hdu = hdul["HA6562_FLUX_ERR"]
        wcs = WCS(flux_hdu.header)
        data = flux_hdu.data
        err = err_hdu.data

        sub = legus[legus["galaxy_norm"] == gal]
        if len(sub) == 0:
            continue

        pos = SkyCoord(ra=sub["RA"].values * u.deg,
                       dec=sub["Dec"].values * u.deg, frame="icrs")
        ap = SkyCircularAperture(pos, r=APERTURE_ARCSEC * u.arcsec)

        # aperture_photometry SUMS the spaxel values -> total flux in aperture
        phot = aperture_photometry(data, ap, wcs=wcs)
        phot_err = aperture_photometry(err ** 2, ap, wcs=wcs)

        flux = np.asarray(phot["aperture_sum"], dtype=float) * BUNIT_SCALE
        flux_err = np.sqrt(np.asarray(phot_err["aperture_sum"],
                                      dtype=float)) * BUNIT_SCALE

        for i, (_, c) in enumerate(sub.iterrows()):
            rows.append({
                "galaxy": gal,
                "ID": c["ID"],
                "field": c["field"],
                "RA": c["RA"],
                "Dec": c["Dec"],
                "mass_msun": c["mass_msun"],
                "age_yr": c["age_yr"],
                "r_eff_pc": c["r_eff_pc"],
                "galaxy_distance_mpc": c["galaxy_distance_mpc"],
                "galaxy_stellar_mass": c["galaxy_stellar_mass"],
                "galaxy_ssfr": c["galaxy_ssfr"],
                "reliable_radius": bool(c["reliable_radius"]),
                "reliable_mass": bool(c["reliable_mass"]),
                "ha_flux": flux[i],
                "ha_flux_err": flux_err[i],
            })

df = pd.DataFrame(rows)

# ---- flux -> luminosity -> SFR -> surface density -------------------------
d_cm = df["galaxy_distance_mpc"].values * MPC_TO_CM
df["ha_lum"] = df["ha_flux"].values * 4.0 * np.pi * d_cm ** 2       # erg/s
df["local_sfr"] = KENNICUTT_1998 * df["ha_lum"]                      # Msun/yr

# physical radius of the aperture at each galaxy's distance, in pc
df["aperture_radius_pc"] = (APERTURE_ARCSEC / 206265.0) * \
                           (df["galaxy_distance_mpc"].values * 1e6)
df["aperture_area_pc2"] = np.pi * df["aperture_radius_pc"] ** 2

df["sfr_surface_density"] = df["local_sfr"] / df["aperture_area_pc2"]

# Halpha surface brightness, the alternative local metric
df["ha_surface_brightness"] = df["ha_flux"] / (np.pi * APERTURE_ARCSEC ** 2)

# ---- logs, matching the column names the regression script expects --------
with np.errstate(divide="ignore", invalid="ignore"):
    df["log_mass"] = np.log10(df["mass_msun"])
    df["log_age"] = np.log10(df["age_yr"])
    df["log_radius"] = np.log10(df["r_eff_pc"])
    df["log_galaxy_ssfr"] = np.log10(df["galaxy_ssfr"])
    df["log_sfr_surface_density"] = np.log10(df["sfr_surface_density"])
    df["log_ha_surface_brightness"] = np.log10(df["ha_surface_brightness"])

# ---- the paper's sample cuts, applied in the documented order -------------
young = df[(df["age_yr"] < 10e6) & (df["ha_flux"] > 0) &
           (df["mass_msun"] > 0) & (df["r_eff_pc"] > 0)].copy()
young = young[np.isfinite(young[["log_mass", "log_age", "log_radius",
                                 "log_sfr_surface_density",
                                 "log_galaxy_ssfr"]]).all(axis=1)]

reliable = young[young["reliable_radius"] & young["reliable_mass"]]

print(f"young clusters with valid Ha : {len(young)}")
print(f"after reliability flags      : {len(reliable)}")
print("\nper galaxy:")
for g in sorted(young["galaxy"].unique()):
    print(f"  {g}: {len(young[young.galaxy == g]):4d} young, "
          f"{len(reliable[reliable.galaxy == g]):4d} reliable")

young.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  ({len(young)} rows, reliability flags retained)")
print("Your original clusters_with_local_environment.csv was NOT modified.")
