from astropy import table
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.signal import savgol_filter

# Add this import at the top of the code
import pandas as pd

# The VIF calculation part (already in the code but needs pandas):
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Set publication-quality style
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Load the catalog data
catalog = table.Table.read("/content/drive/MyDrive/colab_files/LEGUS-sizes/cluster_data.txt", format="ascii.ecsv")

# Parse the LEGUS mass errors
catalog["mass_msun_e-"] = catalog["mass_msun"] - catalog["mass_msun_min"]
catalog["mass_msun_e+"] = catalog["mass_msun_max"] - catalog["mass_msun"]

# Get clusters with reliable radii and masses
mask = catalog["reliable_radius"] & catalog["reliable_mass"]
subset = catalog[mask]

# Prepare data for multivariate regression
# Create feature matrix with log(mass), log(sSFR), log(age)
X_multi = np.column_stack([
    np.log10(subset["mass_msun"]),
    np.log10(subset["galaxy_ssfr"]),  # sSFR from galaxy
    np.log10(subset["age_yr"])
])
y = np.log10(subset["r_eff_pc"])

# Remove any rows with NaN or inf values
valid_mask = np.isfinite(X_multi).all(axis=1) & np.isfinite(y)
X_multi = X_multi[valid_mask]
y = y[valid_mask]
subset_clean = subset[valid_mask]

# Fit the multivariate model
model_multi = LinearRegression()
model_multi.fit(X_multi, y)
y_pred_multi = model_multi.predict(X_multi)

# Calculate metrics
residuals_multi = y - y_pred_multi
r2_multi = r2_score(y, y_pred_multi)
rmse_multi = np.sqrt(mean_squared_error(y, y_pred_multi))
rse_multi = np.std(residuals_multi)
n = len(y)

# Extract parameters
intercept_multi = model_multi.intercept_
coef_mass, coef_ssfr, coef_age = model_multi.coef_

print(f"Multivariate Model: log₁₀(R_eff) = {intercept_multi:.3f} + {coef_mass:.3f}×log₁₀(M) + {coef_ssfr:.4f}×log₁₀(sSFR) + {coef_age:.3f}×log₁₀(Age)")
print(f"R² = {r2_multi:.3f}, RSE = {rse_multi:.3f} dex, n = {n:,} clusters")

# ============= CHART 1: MULTIVARIATE MODEL FIT =============
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Plot actual vs predicted
ax1.scatter(subset_clean["mass_msun"], subset_clean["r_eff_pc"],
            alpha=0.3, s=8, color='#4A90E2',
            label=f'LEGUS clusters (n={n:,})', rasterized=True)

# For visualization, show the relationship with mass while holding other variables at median
median_ssfr = np.median(X_multi[:, 1])
median_age = np.median(X_multi[:, 2])

# Create prediction line at median values
mass_range = [subset_clean["mass_msun"].min(), subset_clean["mass_msun"].max()]
mass_model = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), 200)
log_mass_model = np.log10(mass_model)

# Predict with median values for other variables
X_pred_line = np.column_stack([
    log_mass_model,
    np.full_like(log_mass_model, median_ssfr),
    np.full_like(log_mass_model, median_age)
])
log_radius_pred = model_multi.predict(X_pred_line)
radius_pred = 10**log_radius_pred

# Plot multivariate fit line
ax1.plot(mass_model, radius_pred,
         color='#E74C3C', linewidth=2.5,
         label=f'Multivariate fit (at median sSFR, age)', zorder=5)

# Add prediction bands
radius_upper = 10**(log_radius_pred + rse_multi)
radius_lower = 10**(log_radius_pred - rse_multi)
ax1.fill_between(mass_model, radius_lower, radius_upper,
                  alpha=0.2, color='#E74C3C',
                  label=f'±1σ ({rse_multi:.3f} dex)', zorder=2)

# Set scales and limits
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(1e2, 1e6)
ax1.set_ylim(0.1, 40)

# Labels (no title)
ax1.set_xlabel('Mass [M$_{\odot}$]', fontweight='bold')
ax1.set_ylabel('Effective Radius [pc]', fontweight='bold')

# Add equation box
equation = (f'log$_{{10}}$(R$_{{eff}}$) = {intercept_multi:.3f} + {coef_mass:.3f} log$_{{10}}$(M)\n'
           f'+ {coef_ssfr:.4f} log$_{{10}}$(sSFR) + {coef_age:.3f} log$_{{10}}$(Age)\n'
           f'R² = {r2_multi:.3f}, RSE = {rse_multi:.3f} dex')
ax1.text(0.05, 0.95, equation, transform=ax1.transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3, which='both', linestyle='--')

plt.tight_layout()
plt.show()

# ============= CHART 2: RESIDUALS VS FITTED (MULTIVARIATE) =============
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Plot residuals
ax2.scatter(y_pred_multi, residuals_multi, alpha=0.4, s=8, color='#4A90E2')

# Add reference lines
ax2.axhline(y=0, color='#E74C3C', linestyle='--', linewidth=2, label='Zero residual')
ax2.axhline(y=rse_multi, color='gray', linestyle=':', alpha=0.7, label='±1σ')
ax2.axhline(y=-rse_multi, color='gray', linestyle=':', alpha=0.7)

# Add LOWESS smoothing
sorted_idx = np.argsort(y_pred_multi)
window = min(101, len(residuals_multi)//50*2+1) if len(residuals_multi)//50*2+1 > 5 else 5
if window % 2 == 0:  # Ensure odd window
    window += 1
if len(residuals_multi) > window:
    smoothed = savgol_filter(residuals_multi[sorted_idx], window_length=window, polyorder=3)
    ax2.plot(y_pred_multi[sorted_idx], smoothed,
             color='#F39C12', linewidth=2.5, label='LOWESS smooth')

# Labels (no title)
ax2.set_xlabel('Fitted log₁₀(R$_{eff}$) [Multivariate Model]', fontweight='bold')
ax2.set_ylabel('Residuals [dex]', fontweight='bold')

# Add statistics box
stats_text = (f'Mean: {np.mean(residuals_multi):.3f}\n'
             f'Median: {np.median(residuals_multi):.3f}\n'
             f'Skew: {stats.skew(residuals_multi):.2f}\n'
             f'RSE: {rse_multi:.3f} dex')
ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============= CHART 3: Q-Q PLOT (MULTIVARIATE) =============
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Create Q-Q plot
from scipy import stats
(theoretical_quantiles, sample_quantiles), (slope, intercept, r_value) = stats.probplot(residuals_multi, dist="norm", plot=None)

# Plot the Q-Q points
ax3.scatter(theoretical_quantiles, sample_quantiles, alpha=0.4, s=8, color='#4A90E2', label='Residuals')

# Add the ideal diagonal line (perfect normality)
line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
line_y = slope * line_x + intercept
ax3.plot(line_x, line_y, color='#E74C3C', linewidth=2.5, label='Normal line', zorder=5)

# Add confidence bands for normal distribution (optional but helpful)
# These show where points should fall if truly normal
from scipy.stats import norm
n = len(residuals_multi)
z_critical = norm.ppf(0.975)  # 95% confidence
std_theoretical = np.std(theoretical_quantiles)
se = slope * std_theoretical * np.sqrt(1/n + (theoretical_quantiles - np.mean(theoretical_quantiles))**2 / np.sum((theoretical_quantiles - np.mean(theoretical_quantiles))**2))

# Plot confidence bands
confidence_upper = sample_quantiles[np.argsort(theoretical_quantiles)] + z_critical * se[np.argsort(theoretical_quantiles)]
confidence_lower = sample_quantiles[np.argsort(theoretical_quantiles)] - z_critical * se[np.argsort(theoretical_quantiles)]
sorted_theoretical = np.sort(theoretical_quantiles)

# Simplified confidence bands (straight lines)
ax3.plot(line_x, line_y + z_critical * rse_multi, 'gray', linestyle=':', alpha=0.5)
ax3.plot(line_x, line_y - z_critical * rse_multi, 'gray', linestyle=':', alpha=0.5)

# Labels (no title)
ax3.set_xlabel('Theoretical Quantiles', fontweight='bold')
ax3.set_ylabel('Sample Quantiles', fontweight='bold')

# Add grid for better readability
ax3.grid(True, alpha=0.3)

# Add statistics box
normality_text = (f'Shapiro-Wilk test:\n'
                  f'p-value: {stats.shapiro(residuals_multi[:min(5000, len(residuals_multi))])[1]:.4f}\n'
                  f'Correlation: {r_value:.4f}\n'
                  f'{"Approx. normal" if r_value > 0.98 else "Deviates from normal"}')

ax3.text(0.05, 0.95, normality_text, transform=ax3.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Highlight deviations in tails with colored regions (optional)
# Lower tail deviation
lower_tail_mask = theoretical_quantiles < -2
if np.any(lower_tail_mask):
    ax3.scatter(theoretical_quantiles[lower_tail_mask], sample_quantiles[lower_tail_mask],
                alpha=0.6, s=20, color='#F39C12', label='Tail deviations', zorder=10)

# Upper tail deviation
upper_tail_mask = theoretical_quantiles > 2
if np.any(upper_tail_mask):
    ax3.scatter(theoretical_quantiles[upper_tail_mask], sample_quantiles[upper_tail_mask],
                alpha=0.6, s=20, color='#F39C12', zorder=10)

ax3.legend(loc='lower right')

plt.tight_layout()
plt.show()

#########

# Compare models
print(f"\n{'='*60}")
print("MODEL COMPARISON")
print(f"{'='*60}")
print(f"Mass-only R²: 0.115")  # From  previous analysis
print(f"Multivariate R²: {r2_multi:.3f}")
print(f"Improvement: {(r2_multi - 0.115)*100:.1f}%")
print(f"\nMass-only RSE: 0.272 dex")  # From  previous analysis
print(f"Multivariate RSE: {rse_multi:.3f} dex")
print(f"RSE Reduction: {(0.272 - rse_multi)*1000:.1f} millidex")

# Calculate VIF for multicollinearity check
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Variable"] = ["log(Mass)", "log(sSFR)", "log(Age)"]
vif_data["VIF"] = [variance_inflation_factor(X_multi, i) for i in range(X_multi.shape[1])]

print(f"\n{'='*60}")
print("VARIANCE INFLATION FACTORS")
print(f"{'='*60}")
print(vif_data.to_string(index=False))
