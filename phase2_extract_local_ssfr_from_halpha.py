# PHANGS-MUSE: Extract Local sSFR for LEGUS Clusters
# =======================================================
# This notebook extracts Hα flux at cluster positions and calculates local sSFR
# Date: March 2026

# ==========================================
# SECTION 1: Setup and Mount Google Drive
# ==========================================

from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install astropy photutils -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import SkyCircularAperture, aperture_photometry
import warnings
warnings.filterwarnings('ignore')

print("✅ Setup complete!")

# ==========================================
# SECTION 2: Set File Paths
# ==========================================

# UPDATE THESE PATHS to match your Google Drive folder structure
DRIVE_FOLDER = '/content/drive/MyDrive/colab_files/fits_files'  # Change this!

# LEGUS cluster catalog
LEGUS_FILE = f'{DRIVE_FOLDER}/cluster_sizes_brown_gnedin_21.txt'

# PHANGS Hα maps (extracted files)
MAPS_FILES = {
    'NGC0628': f'{DRIVE_FOLDER}/NGC0628_Ha_only.fits',
    'NGC1433': f'{DRIVE_FOLDER}/NGC1433_Ha_only.fits',
    'NGC1566': f'{DRIVE_FOLDER}/NGC1566_Ha_only.fits',
    'NGC3351': f'{DRIVE_FOLDER}/NGC3351_Ha_only.fits'
}

print(f"LEGUS catalog: {LEGUS_FILE}")
print(f"\nPHANGS Hα maps:")
for gal, path in MAPS_FILES.items():
    print(f"  {gal}: {path}")

# ==========================================
# SECTION 3: Load LEGUS Cluster Catalog
# ==========================================

print("\n" + "="*70)
print("Loading LEGUS cluster catalog...")
print("="*70)

legus = Table.read(LEGUS_FILE, format='ascii.ecsv')

# Normalize galaxy names to match PHANGS
def normalize_galaxy_name(name):
    """Convert ngc628 -> NGC0628"""
    name = str(name).upper()
    if name.startswith('NGC') and len(name) <= 7:
        num_part = name.replace('NGC', '')
        if num_part.isdigit():
            return f"NGC{num_part.zfill(4)}"
    return name

legus['galaxy_normalized'] = [normalize_galaxy_name(g) for g in legus['galaxy']]

# Filter to only galaxies with PHANGS data
phangs_galaxies = list(MAPS_FILES.keys())
clusters_with_phangs = legus[np.isin(legus['galaxy_normalized'], phangs_galaxies)]

print(f"Total LEGUS clusters: {len(legus)}")
print(f"Clusters in PHANGS galaxies: {len(clusters_with_phangs)}")
print("\nBreakdown by galaxy:")
for gal in phangs_galaxies:
    n = len(clusters_with_phangs[clusters_with_phangs['galaxy_normalized'] == gal])
    print(f"  {gal}: {n} clusters")

#=============== check units =========


print("Checking Hα flux units from FITS headers...")
for gal_name, file_path in MAPS_FILES.items():
    hdul = fits.open(file_path)
    ha_hdu = hdul['HA6562_FLUX']

    print(f"\n{gal_name}:")
    print(f"  FITS Header BUNIT: {ha_hdu.header.get('BUNIT', 'NOT FOUND')}")

    # Sample some flux values
    valid_pixels = ha_hdu.data[~np.isnan(ha_hdu.data) & (ha_hdu.data > 0)]
    if len(valid_pixels) > 0:
        print(f"  Sample flux values:")
        print(f"    Min: {valid_pixels.min():.2e}")
        print(f"    Median: {np.median(valid_pixels):.2e}")
        print(f"    Max: {valid_pixels.max():.2e}")

    hdul.close()



# ==========================================
# SECTION 4: Extract Hα Flux at Cluster Positions
# ==========================================

print("\n" + "="*70)
print("Extracting Hα flux at cluster positions...")
print("="*70)

def extract_ha_flux_at_position(ra, dec, ha_map_hdu, method='aperture', aperture_radius=1.0):
    """
    Extract Hα flux at given RA/Dec position

    Parameters:
    -----------
    ra, dec : float
        Cluster position in degrees
    ha_map_hdu : fits HDU
        Hα flux map HDU with WCS
    method : str
        'pixel' = single pixel value
        'aperture' = aperture photometry (more robust)
    aperture_radius : float
        Aperture radius in arcseconds (default 1.0")

    Returns:
    --------
    ha_flux : float
        Hα flux in erg/s/cm²
    """
    wcs = WCS(ha_map_hdu.header)

    if method == 'pixel':
        # Single pixel extraction
        pixel_x, pixel_y = wcs.world_to_pixel_values(ra, dec)
        pixel_x = int(round(pixel_x))
        pixel_y = int(round(pixel_y))

        # Check bounds
        if 0 <= pixel_x < ha_map_hdu.data.shape[1] and 0 <= pixel_y < ha_map_hdu.data.shape[0]:
            return ha_map_hdu.data[pixel_y, pixel_x]
        else:
            return np.nan

    elif method == 'aperture':
        # Aperture photometry (more robust)
        position = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
        aperture = SkyCircularAperture(position, r=aperture_radius*u.arcsec)

        # Perform aperture photometry
        phot_table = aperture_photometry(ha_map_hdu.data, aperture, wcs=wcs)

        # Return sum within aperture
        return phot_table['aperture_sum'][0]

# Process each galaxy
results = []

for gal_name in phangs_galaxies:
    print(f"\n{'='*70}")
    print(f"Processing {gal_name}...")
    print(f"{'='*70}")

    # Load PHANGS Hα map
    hdul = fits.open(MAPS_FILES[gal_name])
    ha_flux_map = hdul['HA6562_FLUX']
    ha_flux_err_map = hdul['HA6562_FLUX_ERR']

    # Get clusters for this galaxy
    gal_clusters = clusters_with_phangs[clusters_with_phangs['galaxy_normalized'] == gal_name]

    print(f"Found {len(gal_clusters)} clusters")
    print(f"Hα map shape: {ha_flux_map.data.shape}")

    # Extract Hα for each cluster
    valid_count = 0
    outside_count = 0
    nan_count = 0

    for cluster in gal_clusters:
        # Extract Hα flux (using aperture method for robustness)
        ha_flux = extract_ha_flux_at_position(
            cluster['RA'],
            cluster['Dec'],
            ha_flux_map,
            method='aperture',
            aperture_radius=1.0  # 1 arcsec aperture
        )

        # Track statistics
        if np.isnan(ha_flux):
            nan_count += 1
        elif ha_flux > 0:
            valid_count += 1

        # Store result
        results.append({
            'galaxy': gal_name,
            'ID': cluster['ID'],
            'RA': cluster['RA'],
            'Dec': cluster['Dec'],
            'mass_msun': cluster['mass_msun'],
            'age_yr': cluster['age_yr'],
            'r_eff_pc': cluster['r_eff_pc'],
            'galaxy_distance_mpc': cluster['galaxy_distance_mpc'],
            'ha_flux': ha_flux,
            'galaxy_stellar_mass': cluster['galaxy_stellar_mass'],
            'galaxy_ssfr': cluster['galaxy_ssfr']
        })

    print(f"  ✅ Valid Hα measurements: {valid_count}")
    print(f"  ⚠️  NaN/invalid: {nan_count}")

    hdul.close()

# Convert to DataFrame
df = pd.DataFrame(results)

print(f"\n{'='*70}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*70}")
print(f"Total clusters processed: {len(df)}")
print(f"Valid Hα measurements: {(df['ha_flux'] > 0).sum()}")
print(f"Invalid/NaN: {(df['ha_flux'].isna() | (df['ha_flux'] <= 0)).sum()}")

# ==========================================
# SECTION 5: Convert Hα Flux to Local SFR
# ==========================================

print("\n" + "="*70)
print("Converting Hα flux to local SFR...")
print("="*70)

def ha_flux_to_sfr(ha_flux, distance_mpc, aperture_radius_arcsec=1.0):
    """
    Convert Hα flux to star formation rate using Kennicutt (1998) calibration

    SFR [Msun/yr] = 5.5e-42 × L(Hα) [erg/s]

    Parameters:
    -----------
    ha_flux : float
        Hα flux in erg/s/cm²/arcsec² (from PHANGS maps)
    distance_mpc : float
        Distance to galaxy in Mpc
    aperture_radius_arcsec : float
        Aperture radius used for extraction

    Returns:
    --------
    sfr : float
        Star formation rate in Msun/yr
    """
    if np.isnan(ha_flux) or ha_flux <= 0:
        return np.nan

    # Aperture area in arcsec²
    aperture_area = np.pi * aperture_radius_arcsec**2

    # Total flux in aperture (erg/s/cm²)
    total_flux = ha_flux * aperture_area

    # Convert distance to cm
    distance_cm = distance_mpc * 3.086e24  # Mpc to cm

    # Calculate luminosity (erg/s)
    luminosity_ha = total_flux * 4 * np.pi * distance_cm**2

    # Kennicutt (1998) calibration
    sfr = 5.5e-42 * luminosity_ha  # Msun/yr

    return sfr

# Calculate local SFR for each cluster
df['local_sfr'] = df.apply(
    lambda row: ha_flux_to_sfr(row['ha_flux'], row['galaxy_distance_mpc']),
    axis=1
)

# Calculate local sSFR
# NOTE: This uses galaxy stellar mass as denominator (approximation)
# Ideally would use local stellar mass from stellar population maps
df['local_ssfr'] = df['local_sfr'] / df['galaxy_stellar_mass']

# Log-transformed versions for regression
df['log_mass'] = np.log10(df['mass_msun'])
df['log_age'] = np.log10(df['age_yr'])
df['log_radius'] = np.log10(df['r_eff_pc'])
df['log_local_ssfr'] = np.log10(df['local_ssfr'])
df['log_galaxy_ssfr'] = np.log10(df['galaxy_ssfr'])

print(f"✅ Calculated local SFR for {(~df['local_sfr'].isna()).sum()} clusters")
print(f"\nLocal SFR statistics (valid measurements only):")
valid_sfr = df[df['local_sfr'] > 0]['local_sfr']
print(f"  Min: {valid_sfr.min():.2e} Msun/yr")
print(f"  Median: {valid_sfr.median():.2e} Msun/yr")
print(f"  Max: {valid_sfr.max():.2e} Msun/yr")

print(f"\nLocal sSFR statistics:")
valid_ssfr = df[df['local_ssfr'] > 0]['local_ssfr']
print(f"  Min: {valid_ssfr.min():.2e} yr⁻¹ ({valid_ssfr.min()*1e9:.2f} Gyr⁻¹)")
print(f"  Median: {valid_ssfr.median():.2e} yr⁻¹ ({valid_ssfr.median()*1e9:.2f} Gyr⁻¹)")
print(f"  Max: {valid_ssfr.max():.2e} yr⁻¹ ({valid_ssfr.max()*1e9:.2f} Gyr⁻¹)")

# ==========================================
# SECTION 6: Quality Checks and Visualization
# ==========================================

print("\n" + "="*70)
print("Data Quality Assessment")
print("="*70)

# Filter to young clusters only (Hα traces recent star formation)
young_mask = df['age_yr'] < 10e6  # Younger than 10 Myr
df['is_young'] = young_mask

print(f"\nYoung clusters (<10 Myr): {young_mask.sum()}")
print(f"Young clusters with valid local sSFR: {(young_mask & (df['local_ssfr'] > 0)).sum()}")

# Plot: Global vs Local sSFR
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Only plot young clusters with valid measurements
plot_data = df[(df['is_young']) & (df['local_ssfr'] > 0)]

# Plot 1: Global vs Local sSFR
axes[0].scatter(plot_data['galaxy_ssfr']*1e9, plot_data['local_ssfr']*1e9,
                alpha=0.5, s=20)
axes[0].plot([0, plot_data['galaxy_ssfr'].max()*1e9], [0, plot_data['galaxy_ssfr'].max()*1e9],
             'r--', label='1:1 line')
axes[0].set_xlabel('Global Galaxy sSFR [Gyr⁻¹]')
axes[0].set_ylabel('Local sSFR [Gyr⁻¹]')
axes[0].set_title('Global vs Local sSFR (Young Clusters)')
axes[0].legend()
axes[0].set_xscale('log')
axes[0].set_yscale('log')

# Plot 2: Local sSFR distribution by galaxy
for gal in phangs_galaxies:
    gal_data = plot_data[plot_data['galaxy'] == gal]
    if len(gal_data) > 0:
        axes[1].hist(np.log10(gal_data['local_ssfr']*1e9), bins=20, alpha=0.5, label=gal)

axes[1].set_xlabel('log₁₀(Local sSFR) [Gyr⁻¹]')
axes[1].set_ylabel('Number of Clusters')
axes[1].set_title('Local sSFR Distribution')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{DRIVE_FOLDER}/local_ssfr_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ Saved diagnostic plot: local_ssfr_diagnostics.png")

# ==========================================
# SECTION 7: Save Final Dataset
# ==========================================

print("\n" + "="*70)
print("Saving final dataset with local sSFR...")
print("="*70)

# Create final analysis dataset (young clusters with valid measurements)
final_dataset = df[
    (df['is_young']) &
    (df['local_ssfr'] > 0) &
    (df['mass_msun'] > 0) &
    (df['r_eff_pc'] > 0)
].copy()

# Select relevant columns
output_columns = [
    'galaxy', 'ID', 'RA', 'Dec',
    'mass_msun', 'age_yr', 'r_eff_pc',
    'ha_flux', 'local_sfr', 'local_ssfr',
    'galaxy_ssfr',  # global sSFR for comparison
    'log_mass', 'log_age', 'log_radius',
    'log_local_ssfr', 'log_galaxy_ssfr'
]

final_dataset = final_dataset[output_columns]

# Save to CSV
output_file = f'{DRIVE_FOLDER}/clusters_with_local_ssfr.csv'
final_dataset.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   {len(final_dataset)} young clusters with valid local sSFR")
print("\nBreakdown by galaxy:")
for gal in phangs_galaxies:
    n = len(final_dataset[final_dataset['galaxy'] == gal])
    print(f"  {gal}: {n} clusters")

# ==========================================
# SECTION 8: Preview Results
# ==========================================

print("\n" + "="*70)
print("SAMPLE DATA (first 5 clusters):")
print("="*70)
print(final_dataset[['galaxy', 'ID', 'mass_msun', 'age_yr', 'r_eff_pc',
                      'local_ssfr', 'galaxy_ssfr']].head())

print("\n" + "="*70)
print("SUMMARY STATISTICS:")
print("="*70)
print(final_dataset[['mass_msun', 'age_yr', 'r_eff_pc', 'local_ssfr', 'galaxy_ssfr']].describe())

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print(f"Final dataset ready for regression analysis:")
print(f"  - {len(final_dataset)} young clusters")
print(f"  - 4 galaxies with PHANGS-MUSE coverage")
print(f"  - Local sSFR from Hα emission")
print(f"\nNext step: Use this dataset to test if local sSFR improves")
print(f"radius predictions beyond mass and age!")
