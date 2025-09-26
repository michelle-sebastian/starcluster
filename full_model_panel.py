import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from astropy import table
import statsmodels.api as sm

#  existing data loading code
catalog = table.Table.read("/content/drive/MyDrive/colab_files/LEGUS-sizes/cluster_data.txt", format="ascii.ecsv")

# Parse the LEGUS mass errors
catalog["mass_msun_e-"] = catalog["mass_msun"] - catalog["mass_msun_min"]
catalog["mass_msun_e+"] = catalog["mass_msun_max"] - catalog["mass_msun"]

# Get the clusters with reliable radii and masses
mask = catalog["reliable_radius"] & catalog["reliable_mass"]
subset = catalog[mask]

# Prepare the data for multivariate linear regression
X_data = np.column_stack((
    np.log10(subset["mass_msun"]),
    np.log10(subset["galaxy_ssfr"]),
    np.log10(subset["age_yr"])
))
y_data = np.log10(subset["r_eff_pc"])

# Create DataFrame
df = pd.DataFrame({
    'log_mass': X_data[:, 0],
    'log_sSFR': X_data[:, 1],
    'log_age': X_data[:, 2],
    'log_radius': y_data
})

# Remove any rows with NaN or infinite values
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Prepare X and y for statsmodels
X = df[['log_mass', 'log_sSFR', 'log_age']]
y = df['log_radius']

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit the multivariate linear regression model
model = sm.OLS(y, X_with_const).fit()

# Get predictions and residuals from the statsmodels model
y_pred = model.fittedvalues
residuals = model.resid
standardized_residuals = residuals / np.std(residuals)
sqrt_abs_std_resid = np.sqrt(np.abs(standardized_residuals))

# Create 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
#fig.suptitle('Full Multivariate Model: Residual Analysis', fontsize=16, y=1.02)

# Color scheme
point_color = '#1f77b4'  # Nice blue
lowess_color = '#ff7f0e'  # Orange
line_color = '#d62728'    # Red

# Panel A: Residuals vs Fitted with LOWESS
ax1 = axes[0, 0]
ax1.scatter(y_pred, residuals, alpha=0.4, s=8, color=point_color, edgecolors='none')
ax1.axhline(y=0, color=line_color, linestyle='--', linewidth=1.5, label='Zero residual')

# Add ±1σ bounds
ax1.axhline(y=np.std(residuals), color='gray', linestyle=':', linewidth=1, alpha=0.5, label='±1σ')
ax1.axhline(y=-np.std(residuals), color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Add LOWESS smooth line
lowess_result = lowess(residuals, y_pred, frac=0.3)
ax1.plot(lowess_result[:, 0], lowess_result[:, 1], color=lowess_color, linewidth=2.5, label='LOWESS smooth')

ax1.set_xlabel('Fitted log₁₀(R_eff) [pc]', fontsize=12)
ax1.set_ylabel('Residuals [dex]', fontsize=12)
ax1.set_title('(A) Residuals vs Fitted Values', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.legend(loc='upper left', fontsize=10)

# Add statistics text box
stats_text = f'Mean: {np.mean(residuals):.4f}\nMedian: {np.median(residuals):.4f}\nRSE: {np.std(residuals):.3f} dex'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Panel B: Q-Q Plot
ax2 = axes[0, 1]
(osm, osr), (slope, intercept, r_value) = stats.probplot(residuals, dist="norm", plot=None)
ax2.scatter(osm, osr, alpha=0.4, s=8, color=point_color, edgecolors='none', label='Residuals')
ax2.plot(osm, slope*osm + intercept, color=line_color, linewidth=1.5, label='Normal line')

# Highlight tail deviations
tail_mask = (np.abs(osm) > 2)
if np.any(tail_mask):
    ax2.scatter(osm[tail_mask], osr[tail_mask], alpha=0.8, s=20, color='orange',
                edgecolors='darkred', label='Tail deviations')

ax2.set_xlabel('Theoretical Quantiles', fontsize=12)
ax2.set_ylabel('Sample Quantiles', fontsize=12)
ax2.set_title('(B) Normal Q-Q Plot', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.legend(loc='upper left', fontsize=10)

# Add Shapiro-Wilk test (use subset if data is too large)
test_sample = residuals[:5000] if len(residuals) > 5000 else residuals
shapiro_stat, shapiro_p = stats.shapiro(test_sample)
qq_text = f'Shapiro-Wilk test:\np-value: {shapiro_p:.5f}\nCorrelation: {r_value:.4f}\nApprox. normal' if shapiro_p > 0.05 else f'Shapiro-Wilk test:\np-value: {shapiro_p:.1e}\nCorrelation: {r_value:.4f}'
ax2.text(0.02, 0.98, qq_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Panel C: Scale-Location Plot
ax3 = axes[1, 0]
ax3.scatter(y_pred, sqrt_abs_std_resid, alpha=0.4, s=8, color=point_color, edgecolors='none')

# Add LOWESS smooth for Scale-Location
lowess_scale = lowess(sqrt_abs_std_resid, y_pred, frac=0.3)
ax3.plot(lowess_scale[:, 0], lowess_scale[:, 1], color=lowess_color, linewidth=2.5, label='LOWESS smooth')

ax3.set_xlabel('Fitted log₁₀(R_eff) [pc]', fontsize=12)
ax3.set_ylabel('√|Standardized Residuals|', fontsize=12)
ax3.set_title('(C) Scale-Location', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax3.legend(loc='upper left', fontsize=10)

# Add note about heteroscedasticity
if np.corrcoef(y_pred, sqrt_abs_std_resid)[0,1] > 0.1:
    ax3.text(0.98, 0.98, 'Possible heteroscedasticity', transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Panel D: Histogram of Residuals
ax4 = axes[1, 1]
n, bins, patches = ax4.hist(residuals, bins=50, density=True, alpha=0.7,
                            color=point_color, edgecolor='black', linewidth=0.5)

# Add normal distribution overlay
mu, sigma = np.mean(residuals), np.std(residuals)
x = np.linspace(residuals.min(), residuals.max(), 200)
ax4.plot(x, stats.norm.pdf(x, mu, sigma), color=line_color, linewidth=2,
         label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

# Add vertical lines for mean and median
ax4.axvline(x=mu, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean = {mu:.4f}')
ax4.axvline(x=np.median(residuals), color='purple', linestyle='--', linewidth=1.5,
            alpha=0.7, label=f'Median = {np.median(residuals):.4f}')

ax4.set_xlabel('Residuals [dex]', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('(D) Distribution of Residuals', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
ax4.legend(loc='upper right', fontsize=9)

# Add statistics text
skewness = stats.skew(residuals)
kurtosis = stats.kurtosis(residuals)
dist_text = f'n = {len(residuals)}\nSkewness: {skewness:.3f}\nKurtosis: {kurtosis:.3f}'
ax4.text(0.02, 0.98, dist_text, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.show()

# Print comprehensive diagnostic statistics
print("=" * 60)
print("FULL MODEL RESIDUAL DIAGNOSTICS SUMMARY")
print("=" * 60)
print(f"\nModel Information:")
print(f"Number of observations: {len(residuals)}")
print(f"Number of predictors: 3 (log_mass, log_sSFR, log_age)")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")

print(f"\nResidual Statistics:")
print(f"Residual Standard Error (RSE): {np.std(residuals):.4f} dex")
print(f"Mean of residuals: {np.mean(residuals):.6f}")
print(f"Median of residuals: {np.median(residuals):.6f}")
print(f"Min residual: {np.min(residuals):.4f}")
print(f"Max residual: {np.max(residuals):.4f}")

print(f"\nNormality Tests:")
print(f"Shapiro-Wilk p-value: {shapiro_p:.4e}")
print(f"Q-Q correlation: {r_value:.4f}")
print(f"Skewness: {skewness:.4f} {'(approx. symmetric)' if abs(skewness) < 0.5 else '(skewed)'}")
print(f"Kurtosis: {kurtosis:.4f} {'(approx. normal)' if abs(kurtosis) < 1 else '(heavy-tailed)' if kurtosis > 1 else '(light-tailed)'}")

# Breusch-Pagan test for heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residuals, X_with_const)
print(f"\nBreusch-Pagan Test for Heteroscedasticity:")
print(f"LM Statistic: {bp_test[0]:.4f}")
print(f"p-value: {bp_test[1]:.4f}")
print(f"Interpretation: {'Evidence of heteroscedasticity (p < 0.05)' if bp_test[1] < 0.05 else 'No strong evidence of heteroscedasticity (p ≥ 0.05)'}")

print("\n" + "=" * 60)

# Save the figure if needed
# fig.savefig('/content/drive/MyDrive/colab_files/full_model_residual_analysis.png', dpi=300, bbox_inches='tight')
