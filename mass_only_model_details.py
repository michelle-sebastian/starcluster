from astropy import table
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the catalog data ( EXACT CODE)
catalog = table.Table.read("/content/drive/MyDrive/colab_files/LEGUS-sizes/cluster_data.txt", format="ascii.ecsv")

# Parse the LEGUS mass errors
catalog["mass_msun_e-"] = catalog["mass_msun"] - catalog["mass_msun_min"]
catalog["mass_msun_e+"] = catalog["mass_msun_max"] - catalog["mass_msun"]

# Get the clusters with reliable radii and masses
mask = catalog["reliable_radius"] & catalog["reliable_mass"]
subset = catalog[mask]

# Prepare the data for linear regression
X_mass = np.log10(subset["mass_msun"]).reshape(-1, 1)  # Independent variable (log of mass)
y = np.log10(subset["r_eff_pc"])  # Dependent variable (log of radius)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_mass, y)

# Get predictions and calculate metrics
y_pred = model.predict(X_mass)
residuals = y - y_pred

# Calculate model metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
rse = np.std(residuals)
n = len(y)

# Extract model parameters
intercept = model.intercept_
slope = model.coef_[0]

print("=" * 70)
print(f"MASS-ONLY MODEL RESULTS - LEGUS CATALOG")
print("=" * 70)
print(f"Dataset Size: {n:,} clusters with reliable measurements")
print(f"\nFitted Equation: log₁₀(R_eff) = {intercept:.4f} + {slope:.4f} × log₁₀(Mass)")
print(f"Power Law Form: R_eff ≈ {10**intercept:.4f} × M^{slope:.4f}")
print("\nModel Performance Metrics:")
print(f"  • R²: {r2:.4f}")
print(f"  • RMSE: {rmse:.4f} dex")
print(f"  • MAE: {mae:.4f} dex")
print(f"  • Residual Std Error: {rse:.4f} dex")
print("=" * 70)

# Create the main plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot all data points with error bars
ax.errorbar(subset["mass_msun"], subset["r_eff_pc"],
            xerr=[subset["mass_msun_e-"], subset["mass_msun_e+"]],
            yerr=[subset["r_eff_pc_e-"], subset["r_eff_pc_e+"]],
            fmt='o', markersize=3, alpha=0.3, color='#2E86AB',
            ecolor='gray', elinewidth=0.5, capsize=0,
            label=f'LEGUS Clusters (n={n:,})', rasterized=True)

# Create model line
mass_range = [subset["mass_msun"].min(), subset["mass_msun"].max()]
log_mass_range = np.log10(mass_range)
mass_model = np.logspace(log_mass_range[0], log_mass_range[1], 200)
log_mass_model = np.log10(mass_model)
log_radius_model = intercept + slope * log_mass_model
radius_model = 10**log_radius_model

# Plot the best-fit line
ax.plot(mass_model, radius_model,
        color='#A23B72', linewidth=3,
        label=f'Best Fit: R_eff ∝ M^{slope:.3f}', zorder=5)

# Add 1-sigma prediction bands (68% confidence)
radius_upper_1sig = 10**(log_radius_model + rse)
radius_lower_1sig = 10**(log_radius_model - rse)

ax.fill_between(mass_model, radius_lower_1sig, radius_upper_1sig,
                 alpha=0.3, color='#F18F01',
                 label=f'±1σ ({rse:.3f} dex)', zorder=2)

# Add 2-sigma prediction bands (95% confidence)
radius_upper_2sig = 10**(log_radius_model + 2*rse)
radius_lower_2sig = 10**(log_radius_model - 2*rse)

ax.fill_between(mass_model, radius_lower_2sig, radius_upper_2sig,
                 alpha=0.15, color='#C73E1D',
                 label='±2σ (95% CI)', zorder=1)

# Set log scales
ax.set_xscale('log')
ax.set_yscale('log')

# Set axis limits based on Brown & Gnedin (2021) paper
ax.set_xlim(1e2, 1e6)
ax.set_ylim(0.1, 40)

# Labels and title
ax.set_xlabel('Mass [M$_{\odot}$]', fontsize=14, fontweight='bold')
ax.set_ylabel('Radius [pc]', fontsize=14, fontweight='bold')
#ax.set_title('Star Cluster Radius-Mass Relation (LEGUS Survey)', fontsize=16, fontweight='bold', pad=20)

# Add equation and statistics text box
equation_text = f'$\\log_{{10}}(R_{{\\rm eff}}) = {intercept:.3f} + {slope:.3f} \\times \\log_{{10}}(M)$'
stats_text = (f'R² = {r2:.3f}\n'
              f'RMSE = {rmse:.3f} dex\n'
              f'RSE = {rse:.3f} dex\n'
              f'n = {n:,} clusters')
textbox = equation_text + '\n' + stats_text

ax.text(0.05, 0.95, textbox, transform=ax.transAxes,
        fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Customize grid
ax.grid(True, which='both', alpha=0.3, linestyle='--')
ax.grid(True, which='major', alpha=0.5)

# Legend
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

# Adjust layout
plt.tight_layout()
plt.show()

# Create comprehensive residual analysis
fig2, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Residuals vs Fitted Values
ax1 = axes[0, 0]
ax1.scatter(y_pred, residuals, alpha=0.4, s=10, color='#2E86AB')
ax1.axhline(y=0, color='#A23B72', linestyle='--', linewidth=2)
ax1.axhline(y=rse, color='gray', linestyle=':', alpha=0.7, label='±1σ')
ax1.axhline(y=-rse, color='gray', linestyle=':', alpha=0.7)
ax1.axhline(y=2*rse, color='lightgray', linestyle=':', alpha=0.5, label='±2σ')
ax1.axhline(y=-2*rse, color='lightgray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Fitted log₁₀(R_eff)', fontsize=11)
ax1.set_ylabel('Residuals [dex]', fontsize=11)
ax1.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)

# Add LOWESS smoothing
from scipy.signal import savgol_filter
sorted_idx = np.argsort(y_pred)
window_length = min(101, max(5, len(residuals)//50)*2+1)  # Ensure odd window
if len(residuals) > window_length:
    smoothed = savgol_filter(residuals[sorted_idx], window_length=window_length, polyorder=3)
    ax1.plot(y_pred[sorted_idx], smoothed, color='red', linewidth=2, alpha=0.7, label='LOWESS')

# 2. Q-Q plot for normality
ax2 = axes[0, 1]
stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
ax2.set_xlabel('Theoretical Quantiles', fontsize=11)
ax2.set_ylabel('Sample Quantiles', fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. Scale-Location plot
ax3 = axes[0, 2]
standardized_residuals = residuals / rse
sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
ax3.scatter(y_pred, sqrt_abs_resid, alpha=0.4, s=10, color='#2E86AB')
if len(residuals) > window_length:
    smoothed_sqrt = savgol_filter(sqrt_abs_resid[sorted_idx], window_length=window_length, polyorder=3)
    ax3.plot(y_pred[sorted_idx], smoothed_sqrt, color='red', linewidth=2, alpha=0.7)
ax3.set_xlabel('Fitted log₁₀(R_eff)', fontsize=11)
ax3.set_ylabel('√|Standardized Residuals|', fontsize=11)
ax3.set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Histogram of residuals
ax4 = axes[1, 0]
n_bins = min(50, max(20, int(np.sqrt(n))))
counts, bins, patches = ax4.hist(residuals, bins=n_bins, density=True, alpha=0.7,
                                  color='#2E86AB', edgecolor='black')
# Overlay normal distribution
x_norm = np.linspace(residuals.min(), residuals.max(), 100)
ax4.plot(x_norm, stats.norm.pdf(x_norm, 0, rse), 'r-', linewidth=2, label=f'N(0, {rse:.3f})')
ax4.set_xlabel('Residuals [dex]', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Residuals vs Log(Mass)
ax5 = axes[1, 1]
ax5.scatter(X_mass.flatten(), residuals, alpha=0.4, s=10, color='#2E86AB')
ax5.axhline(y=0, color='#A23B72', linestyle='--', linewidth=2)
ax5.axhline(y=rse, color='gray', linestyle=':', alpha=0.7)
ax5.axhline(y=-rse, color='gray', linestyle=':', alpha=0.7)
ax5.set_xlabel('log₁₀(Mass/M$_{\odot}$)', fontsize=11)
ax5.set_ylabel('Residuals [dex]', fontsize=11)
ax5.set_title('Residuals vs log(Mass)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Cook's Distance
ax6 = axes[1, 2]
# Calculate Cook's distance
n_params = 2  # intercept and slope
leverage = 1/n + (X_mass.flatten() - X_mass.mean())**2 / np.sum((X_mass.flatten() - X_mass.mean())**2)
cooks_d = (residuals**2 / (n_params * rse**2)) * (leverage / (1 - leverage)**2)
ax6.stem(range(len(cooks_d)), cooks_d, basefmt=' ', markerfmt='o', linefmt='b-')
ax6.axhline(y=4/n, color='red', linestyle='--', label=f'4/n threshold')
ax6.set_xlabel('Observation Index', fontsize=11)
ax6.set_ylabel("Cook's Distance", fontsize=11)
ax6.set_title("Cook's Distance (Influential Points)", fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle(f'Residual Diagnostic Plots (n={n:,} clusters)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Statistical tests
print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

# Shapiro-Wilk test (use subsample if needed)
sample_size = min(5000, len(residuals))
test_residuals = residuals if len(residuals) <= 5000 else np.random.choice(residuals, 5000, replace=False)
shapiro_stat, shapiro_p = stats.shapiro(test_residuals)
print(f"\nShapiro-Wilk Test for Normality (n={sample_size}):")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
print(f"  Conclusion: {'Residuals are normally distributed' if shapiro_p > 0.05 else 'Residuals deviate from normality'} (α=0.05)")

# Breusch-Pagan test
from scipy.stats import chi2
residuals_squared = residuals**2
bp_model = LinearRegression()
bp_model.fit(X_mass, residuals_squared)
bp_pred = bp_model.predict(X_mass)
bp_r2 = r2_score(residuals_squared, bp_pred)
bp_stat = n * bp_r2
bp_p = 1 - chi2.cdf(bp_stat, df=1)
print(f"\nBreusch-Pagan Test for Heteroscedasticity:")
print(f"  Statistic: {bp_stat:.4f}")
print(f"  p-value: {bp_p:.4f}")
print(f"  Conclusion: {'Constant variance (homoscedastic)' if bp_p > 0.05 else 'Non-constant variance (heteroscedastic)'} (α=0.05)")

# Durbin-Watson test
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(residuals)
print(f"\nDurbin-Watson Test for Autocorrelation:")
print(f"  Statistic: {dw_stat:.4f}")
if dw_stat < 1.5:
    print(f"  Conclusion: Positive autocorrelation detected")
elif dw_stat > 2.5:
    print(f"  Conclusion: Negative autocorrelation detected")
else:
    print(f"  Conclusion: No significant autocorrelation")

# Summary statistics
print("\n" + "=" * 70)
print("RESIDUAL SUMMARY STATISTICS")
print("=" * 70)
print(f"  Mean: {np.mean(residuals):.6f}")
print(f"  Median: {np.median(residuals):.4f}")
print(f"  Std Dev: {np.std(residuals):.4f}")
print(f"  Min: {np.min(residuals):.4f}")
print(f"  Max: {np.max(residuals):.4f}")
print(f"  Skewness: {stats.skew(residuals):.4f}")
print(f"  Kurtosis: {stats.kurtosis(residuals):.4f}")

# Print predictions at key masses
print("\n" + "=" * 70)
print("MODEL PREDICTIONS AT REPRESENTATIVE MASSES")
print("=" * 70)
for log_m in [3, 4, 5, 6]:
    m = 10**log_m
    log_r_pred = intercept + slope * log_m
    r_pred = 10**log_r_pred
    r_lower_68 = 10**(log_r_pred - rse)
    r_upper_68 = 10**(log_r_pred + rse)
    r_lower_95 = 10**(log_r_pred - 2*rse)
    r_upper_95 = 10**(log_r_pred + 2*rse)
    print(f"\nM = 10^{log_m} M☉:")
    print(f"  Predicted R_eff = {r_pred:.3f} pc")
    print(f"  68% CI: [{r_lower_68:.3f}, {r_upper_68:.3f}] pc")
    print(f"  95% CI: [{r_lower_95:.3f}, {r_upper_95:.3f}] pc")
print("=" * 70)
