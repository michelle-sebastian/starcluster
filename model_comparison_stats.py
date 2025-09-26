import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# First, check for missing values in  data
print("Checking for missing values...")
print(f"Missing values in mass: {np.sum(np.isnan(subset['mass_msun']))}")
print(f"Missing values in sSFR: {np.sum(np.isnan(subset['galaxy_ssfr']))}")
print(f"Missing values in age: {np.sum(np.isnan(subset['age_yr']))}")
print(f"Missing values in radius: {np.sum(np.isnan(subset['r_eff_pc']))}")

# Remove any rows with missing values
# Create mask for non-missing values
valid_mask = (
    ~np.isnan(subset["mass_msun"]) &
    ~np.isnan(subset["galaxy_ssfr"]) &
    ~np.isnan(subset["age_yr"]) &
    ~np.isnan(subset["r_eff_pc"]) &
    (subset["mass_msun"] > 0) &  # Ensure positive values for log
    (subset["galaxy_ssfr"] > 0) &
    (subset["age_yr"] > 0) &
    (subset["r_eff_pc"] > 0)
)

# Apply the mask
subset_clean = subset[valid_mask]
print(f"\nOriginal subset size: {len(subset)}")
print(f"After removing missing/invalid values: {len(subset_clean)}")

# Recreate X and y with clean data
X = np.column_stack((
    np.log10(subset_clean["mass_msun"]),
    np.log10(subset_clean["galaxy_ssfr"]),
    np.log10(subset_clean["age_yr"])
))
y = np.log10(subset_clean["r_eff_pc"])

# Check for any remaining NaN or infinite values
print(f"\nAny NaN in X: {np.any(np.isnan(X))}")
print(f"Any infinite in X: {np.any(np.isinf(X))}")
print(f"Any NaN in y: {np.any(np.isnan(y))}")
print(f"Any infinite in y: {np.any(np.isinf(y))}")

# If there are still issues, remove them
mask_final = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
X_clean = X[mask_final]
y_clean = y[mask_final]

print(f"Final sample size: {len(X_clean)}")

# Now proceed with the analysis using X_clean and y_clean
# =========================================================

# Table 1 - Regression Coefficients
print("\n" + "="*50)
print("TABLE 1: REGRESSION COEFFICIENTS")
print("="*50)

# Fit model with statsmodels for detailed statistics
X_with_const = sm.add_constant(X_clean)
model_sm = sm.OLS(y_clean, X_with_const).fit()
print(model_sm.summary())

# Extract key values
coefficients = model_sm.params
std_errors = model_sm.bse
t_values = model_sm.tvalues
p_values = model_sm.pvalues
conf_int = model_sm.conf_int()

print("\nFormatted for paper:")
print(f"Intercept: β = {coefficients[0]:.4f}, SE = {std_errors[0]:.4f}, p = {p_values[0]:.4f}")
print(f"log(Mass): β = {coefficients[1]:.4f}, SE = {std_errors[1]:.4f}, p = {p_values[1]:.4f}")
print(f"log(sSFR): β = {coefficients[2]:.4f}, SE = {std_errors[2]:.4f}, p = {p_values[2]:.4f}")
print(f"log(Age):  β = {coefficients[3]:.4f}, SE = {std_errors[3]:.4f}, p = {p_values[3]:.4f}")
print(f"\nModel R² = {model_sm.rsquared:.4f}")
print(f"Adjusted R² = {model_sm.rsquared_adj:.4f}")
print(f"RSE = {np.sqrt(model_sm.mse_resid):.4f}")

# Table 2 - VIF Calculation
print("\n" + "="*50)
print("TABLE 2: MULTICOLLINEARITY DIAGNOSTICS")
print("="*50)

# Create DataFrame for VIF calculation
X_df = pd.DataFrame(X_clean, columns=['log_Mass', 'log_sSFR', 'log_Age'])

# Calculate VIF - Alternative method that handles issues better
from sklearn.linear_model import LinearRegression

vif_values = []
for i in range(X_df.shape[1]):
    # Get all columns except i
    X_temp = X_df.drop(X_df.columns[i], axis=1)
    y_temp = X_df.iloc[:, i]

    # Fit regression
    reg = LinearRegression()
    reg.fit(X_temp, y_temp)
    r_squared = reg.score(X_temp, y_temp)

    # Calculate VIF
    vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
    vif_values.append(vif)

# Create VIF table
vif_data = pd.DataFrame({
    'Variable': X_df.columns,
    'VIF': vif_values,
    'Tolerance': [1/v if v != np.inf else 0 for v in vif_values]
})
print("\nVIF Values:")
print(vif_data)

# Correlation matrix
corr_matrix = X_df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Table 3 - Model Comparison
print("\n" + "="*50)
print("TABLE 3: MODEL COMPARISON")
print("="*50)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

# Full model with sklearn
model_full = LinearRegression()
model_full.fit(X_clean, y_clean)
y_pred_full = model_full.predict(X_clean)
r2_full = r2_score(y_clean, y_pred_full)
rmse_full = np.sqrt(mean_squared_error(y_clean, y_pred_full))
mae_full = mean_absolute_error(y_clean, y_pred_full)

# Cross-validation for full model
cv_scores_full = cross_val_score(model_full, X_clean, y_clean, cv=5, scoring='r2')
cv_r2_full = cv_scores_full.mean()
cv_rmse_full_scores = cross_val_score(model_full, X_clean, y_clean, cv=5,
                                       scoring='neg_mean_squared_error')
cv_rmse_full = np.sqrt(-cv_rmse_full_scores.mean())

# Mass-only model
X_mass_only = X_clean[:, 0].reshape(-1, 1)
model_mass = LinearRegression()
model_mass.fit(X_mass_only, y_clean)
y_pred_mass = model_mass.predict(X_mass_only)
r2_mass = r2_score(y_clean, y_pred_mass)
rmse_mass = np.sqrt(mean_squared_error(y_clean, y_pred_mass))
mae_mass = mean_absolute_error(y_clean, y_pred_mass)

# Cross-validation for mass-only model
cv_scores_mass = cross_val_score(model_mass, X_mass_only, y_clean, cv=5, scoring='r2')
cv_r2_mass = cv_scores_mass.mean()
cv_rmse_mass_scores = cross_val_score(model_mass, X_mass_only, y_clean, cv=5,
                                       scoring='neg_mean_squared_error')
cv_rmse_mass = np.sqrt(-cv_rmse_mass_scores.mean())

print(f"\nFull Model:")
print(f"  R² = {r2_full:.4f}, Adj R² = {model_sm.rsquared_adj:.4f}")
print(f"  RMSE = {rmse_full:.4f}, MAE = {mae_full:.4f}")
print(f"  CV R² = {cv_r2_full:.4f}, CV RMSE = {cv_rmse_full:.4f}")

print(f"\nMass-Only Model:")
print(f"  R² = {r2_mass:.4f}")
print(f"  RMSE = {rmse_mass:.4f}, MAE = {mae_mass:.4f}")
print(f"  CV R² = {cv_r2_mass:.4f}, CV RMSE = {cv_rmse_mass:.4f}")

# F-test for nested models
n = len(y_clean)
k_full = 4  # intercept + 3 predictors
k_reduced = 2  # intercept + 1 predictor

f_stat = ((r2_full - r2_mass) / (k_full - k_reduced)) / ((1 - r2_full) / (n - k_full))
f_pvalue = 1 - stats.f.cdf(f_stat, k_full - k_reduced, n - k_full)
print(f"\nF-test (Full vs Mass-Only):")
print(f"  F({k_full - k_reduced}, {n - k_full}) = {f_stat:.3f}, p = {f_pvalue:.6f}")

# Calculate AIC and BIC
def calculate_aic_bic(n, k, rss):
    """Calculate AIC and BIC from residual sum of squares"""
    aic = n * np.log(rss/n) + 2*k
    bic = n * np.log(rss/n) + k*np.log(n)
    return aic, bic

# RSS for each model
rss_full = np.sum((y_clean - y_pred_full)**2)
rss_mass = np.sum((y_clean - y_pred_mass)**2)

aic_full, bic_full = calculate_aic_bic(n, k_full, rss_full)
aic_mass, bic_mass = calculate_aic_bic(n, k_reduced, rss_mass)

print(f"\nInformation Criteria:")
print(f"  Full Model: AIC = {aic_full:.1f}, BIC = {bic_full:.1f}")
print(f"  Mass-Only:  AIC = {aic_mass:.1f}, BIC = {bic_mass:.1f}")
print(f"  ΔAIC = {aic_mass - aic_full:.1f} (negative favors full model)")
print(f"  ΔBIC = {bic_mass - bic_full:.1f} (negative favors full model)")
