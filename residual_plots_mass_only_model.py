import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def generate_residual_plots_astropy(astropy_table, figsize=(15, 12)):
    """
    Generate residual diagnostic plots for Astropy Table data

    Parameters:
    -----------
    astropy_table : astropy.table.Table
        Astropy table with mass_msun and r_eff_pc columns
    figsize : tuple
        Figure size for the plots
    """

    print("=== ASTROPY TABLE RESIDUAL ANALYSIS ===")
    print(f"Table type: {type(astropy_table)}")
    print(f"Table length: {len(astropy_table)}")
    print(f"Available columns: {astropy_table.colnames}")

    # Extract data from Astropy Table (correct method)
    try:
        # For Astropy tables, use .data or direct numpy conversion
        mass_data = np.array(astropy_table['mass_msun'])
        radius_data = np.array(astropy_table['r_eff_pc'])
        print("✓ Successfully extracted data from Astropy Table")
    except KeyError as e:
        print(f"❌ Column not found: {e}")
        print(f"Available columns: {astropy_table.colnames}")
        return None
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        return None

    # Check for invalid values
    valid_mask = (mass_data > 0) & (radius_data > 0) & np.isfinite(mass_data) & np.isfinite(radius_data)
    mass_clean = mass_data[valid_mask]
    radius_clean = radius_data[valid_mask]

    print(f"Original data points: {len(mass_data)}")
    print(f"After removing invalid values: {len(mass_clean)}")
    print(f"Mass range: {mass_clean.min():.2e} to {mass_clean.max():.2e} solar masses")
    print(f"Radius range: {radius_clean.min():.3f} to {radius_clean.max():.3f} pc")

    # Apply log transformation
    log_mass = np.log10(mass_clean)
    log_radius = np.log10(radius_clean)

    # Fit mass-only linear regression
    X = log_mass.reshape(-1, 1)
    y = log_radius

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Calculate model statistics
    r2 = r2_score(y, y_pred)
    residual_std_error = np.std(residuals)
    n_params = 2  # intercept + slope

    print(f"\n=== MODEL STATISTICS ===")
    print(f"Equation: log₁₀(R_eff) = {model.intercept_:.3f} + {model.coef_[0]:.3f} × log₁₀(Mass)")
    print(f"R² = {r2:.4f}")
    print(f"Residual Standard Error: {residual_std_error:.4f} dex")
    print(f"Sample size: {len(residuals)}")

    # Calculate standardized residuals
    residual_std = residuals / residual_std_error

    # Calculate leverage (for simple linear regression)
    X_with_intercept = np.column_stack([np.ones(len(X)), X.flatten()])
    try:
        hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
        leverage = np.diag(hat_matrix)

        # Calculate Cook's distance
        mse = np.sum(residuals**2) / (len(residuals) - n_params)
        cooks_d = (residual_std**2 / n_params) * (leverage / (1 - leverage)**2)
    except:
        # Fallback if matrix operations fail
        leverage = np.ones(len(residuals)) / len(residuals)
        cooks_d = residual_std**2 / n_params

    # Create the four diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Residual Diagnostic Plots: Mass-Only Model',
                 fontsize=16, fontweight='bold')

    # Plot (a): Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_pred, residuals, alpha=0.6, s=20, color='steelblue', edgecolor='none')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)

    # Add LOWESS smooth line
    try:
        from scipy.interpolate import interp1d
        # Simple polynomial fit for trend
        sorted_idx = np.argsort(y_pred)
        z = np.polyfit(y_pred[sorted_idx], residuals[sorted_idx], 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(y_pred.min(), y_pred.max(), 100)
        ax1.plot(x_smooth, p(x_smooth), color='red', linewidth=2, alpha=0.8)
    except:
        pass

    ax1.set_xlabel('Fitted Values [log₁₀(R_eff)]', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('(a) Residuals vs Fitted', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add statistics box
    ax1.text(0.02, 0.98, f'RSE: {residual_std_error:.3f} dex\nR² = {r2:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot (b): Q-Q Plot
    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('(b) Normal Q-Q Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    ax2.text(0.02, 0.98, f'Shapiro-Wilk Test:\nW = {shapiro_stat:.3f}\np = {shapiro_p:.3f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot (c): Scale-Location
    ax3 = axes[1, 0]
    sqrt_abs_std_resid = np.sqrt(np.abs(residual_std))
    ax3.scatter(y_pred, sqrt_abs_std_resid, alpha=0.6, s=20, color='green', edgecolor='none')

    # Trend line
    try:
        z = np.polyfit(y_pred, sqrt_abs_std_resid, 1)
        p = np.poly1d(z)
        x_smooth = np.linspace(y_pred.min(), y_pred.max(), 100)
        ax3.plot(x_smooth, p(x_smooth), color='red', linewidth=2, alpha=0.8)
        slope = z[0]
    except:
        slope = 0

    ax3.set_xlabel('Fitted Values [log₁₀(R_eff)]', fontsize=12)
    ax3.set_ylabel('√|Standardized Residuals|', fontsize=12)
    ax3.set_title('(c) Scale-Location Plot', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Homoscedasticity assessment
    if abs(slope) < 0.1:
        homoscedasticity = "Good"
    else:
        homoscedasticity = "Questionable"

    ax3.text(0.02, 0.98, f'Homoscedasticity:\n{homoscedasticity}\nSlope: {slope:.3f}',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot (d): Residuals vs Leverage
    ax4 = axes[1, 1]
    ax4.scatter(leverage, residual_std, alpha=0.6, s=20, color='purple', edgecolor='none')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.axhline(y=2, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='±2σ')
    ax4.axhline(y=-2, color='orange', linestyle='--', linewidth=1, alpha=0.7)

    # Cook's distance contours
    lev_range = np.linspace(leverage.min(), max(leverage.max(), 0.01), 100)
    try:
        cooks_05 = np.sqrt(0.5 * n_params * (1 - lev_range) / lev_range)
        cooks_1 = np.sqrt(1.0 * n_params * (1 - lev_range) / lev_range)

        ax4.plot(lev_range, cooks_05, 'r--', alpha=0.5, linewidth=1, label="Cook's D = 0.5")
        ax4.plot(lev_range, -cooks_05, 'r--', alpha=0.5, linewidth=1)
        ax4.plot(lev_range, cooks_1, 'r-', alpha=0.7, linewidth=1, label="Cook's D = 1.0")
        ax4.plot(lev_range, -cooks_1, 'r-', alpha=0.7, linewidth=1)
    except:
        pass

    ax4.set_xlabel('Leverage', fontsize=12)
    ax4.set_ylabel('Standardized Residuals', fontsize=12)
    ax4.set_title('(d) Residuals vs Leverage', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10)

    # Identify problematic points
    high_leverage = leverage > (2 * n_params / len(leverage))
    high_cooks = cooks_d > (4 / len(cooks_d))
    outliers = np.abs(residual_std) > 2

    ax4.text(0.02, 0.98, f'High Leverage: {np.sum(high_leverage)}\nHigh Cook\'s D: {np.sum(high_cooks)}\nOutliers (|z|>2): {np.sum(outliers)}',
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Print diagnostic summary
    print(f"\n=== DIAGNOSTIC SUMMARY ===")
    print(f"Normality (Shapiro-Wilk): p = {shapiro_p:.4f} {'✓' if shapiro_p > 0.05 else '⚠'}")
    print(f"Homoscedasticity: {homoscedasticity}")
    print(f"High leverage points: {np.sum(high_leverage)} ({100*np.sum(high_leverage)/len(leverage):.1f}%)")
    print(f"High influence points: {np.sum(high_cooks)} ({100*np.sum(high_cooks)/len(cooks_d):.1f}%)")
    print(f"Outliers (|z|>2): {np.sum(outliers)} ({100*np.sum(outliers)/len(residual_std):.1f}%)")

    # Overall model adequacy
    adequacy_checks = [
        shapiro_p > 0.05,  # Normality
        abs(slope) < 0.1,   # Homoscedasticity
        np.sum(high_cooks) < len(cooks_d) * 0.05,  # Few influential points
        np.sum(outliers) < len(residual_std) * 0.05   # Few outliers
    ]

    adequacy_score = sum(adequacy_checks)
    print(f"\nModel Adequacy Score: {adequacy_score}/4")

    if adequacy_score >= 3:
        print("✓ Model assumptions appear well satisfied")
    elif adequacy_score >= 2:
        print("⚠ Some concerns with model assumptions")
    else:
        print("❌ Significant concerns with model assumptions")

    return {
        'residuals': residuals,
        'fitted': y_pred,
        'leverage': leverage,
        'cooks_distance': cooks_d,
        'shapiro_p': shapiro_p,
        'residual_std_error': residual_std_error,
        'r2': r2,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'adequacy_score': adequacy_score
    }

# Usage for  Astropy Table:
print("For Astropy Table, run:")
print("results = generate_residual_plots_astropy(subset_clean)")
generate_residual_plots_astropy(subset_clean)
print("plt.show()")
