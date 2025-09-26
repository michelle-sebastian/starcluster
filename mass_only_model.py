from astropy import table
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


#location of the catalog data file
#https://github.com/gillenbrown/LEGUS-sizes/blob/master/cluster_sizes_brown_gnedin_21.txt

# Load the catalog data
catalog = table.Table.read("/content/drive/MyDrive/colab_files/LEGUS-sizes/cluster_data.txt", format="ascii.ecsv")

# Parse the LEGUS mass errors
catalog["mass_msun_e-"] = catalog["mass_msun"] - catalog["mass_msun_min"]
catalog["mass_msun_e+"] = catalog["mass_msun_max"] - catalog["mass_msun"]

# Get the clusters with reliable radii and masses
mask = catalog["reliable_radius"] & catalog["reliable_mass"]
subset = catalog[mask]

# Prepare the data for linear regression
X = np.log10(subset["mass_msun"]).reshape(-1, 1)  # Independent variable (log of mass)
y = np.log10(subset["r_eff_pc"])  # Dependent variable (log of radius)

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the radii using the fitted model
y_pred = model.predict(X)

# Plot the data
fig, ax = plt.subplots()
ax.errorbar(
    x=subset["mass_msun"],
    y=subset["r_eff_pc"],
    fmt="o",
    markersize=2,
    lw=0.3,
    xerr=[subset["mass_msun_e-"], subset["mass_msun_e+"]],
    yerr=[subset["r_eff_pc_e-"], subset["r_eff_pc_e+"]],
    label="Data"
)

# Plot the regression line
ax.plot(subset["mass_msun"], 10**y_pred, color="red", label="Linear Regression Fit")

# Plot formatting
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e6)
ax.set_ylim(0.1, 40)
ax.set_xlabel("Mass [$M_\odot$]")
ax.set_ylabel("Radius [pc]")
ax.legend()
plt.show()

# Print the slope and intercept of the regression line
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
