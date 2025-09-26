# Star Cluster Radius Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the analysis code for investigating the radius-mass relationship of star clusters using the LEGUS (Legacy ExtraGalactic UV Survey) catalog. The project implements both mass-only and multivariate regression models to understand how star cluster effective radii depend on mass, specific star formation rate (sSFR), and age.

## Key Features

- **Mass-only model**: Simple power-law relationship between cluster radius and mass
- **Multivariate model**: Incorporates mass, galaxy sSFR, and cluster age
- **Comprehensive diagnostics**: Residual analysis, normality tests, heteroscedasticity checks
- **Publication-ready visualizations**: Automated generation of all figures

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/michelle-sebastian/starcluster.git
cd starcluster
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Run the complete analysis:
```bash
python scripts/reproduce_results.py
```

### Run individual models:
```bash
# Mass-only model
python scripts/run_mass_only.py

# Multivariate model
python scripts/run_multivariate.py
```

### Generate publication figures:
```bash
python scripts/generate_all_figures.py
```

## Project Structure

```
starcluster/
├── data/               # Data files (not tracked)
├── src/                # Source code
│   ├── models/         # Model implementations
│   ├── analysis/       # Statistical analysis
│   └── visualization/  # Plotting functions
├── notebooks/          # Jupyter notebooks
├── scripts/            # Executable scripts
└── results/            # Output figures and tables
```

## Data

The analysis uses the LEGUS star cluster catalog. Due to size constraints, the data file is not included in this repository. 

### Obtaining the Data

Download the LEGUS catalog from:
- [Brown & Gnedin (2021) catalog](https://github.com/gillenbrown/LEGUS-sizes/blob/master/cluster_sizes_brown_gnedin_21.txt)

Place the file in `data/raw/cluster_data.txt`

## Models

### Mass-Only Model
```
log₁₀(R_eff) = α + β × log₁₀(M)
```
- Simple power-law relationship
- R² ≈ 0.115
- RSE ≈ 0.272 dex

### Multivariate Model
```
log₁₀(R_eff) = α + β₁×log₁₀(M) + β₂×log₁₀(sSFR) + β₃×log₁₀(Age)
```
- Includes environmental and evolutionary factors
- R² ≈ 0.140
- RSE ≈ 0.267 dex

## Results

Key findings:
- Cluster radius scales with mass as R ∝ M^0.2
- Weak correlation with galaxy sSFR
- Age dependence suggests evolutionary expansion
- Significant scatter indicates additional unmodeled physics

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sebastian2024starcluster,
  author = {Sebastian, Michelle},
  title = {Star Cluster Radius Analysis},
  year = {2024},
  url = {https://github.com/michelle-sebastian/starcluster}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Michelle Sebastian - [GitHub](https://github.com/michelle-sebastian)

## Acknowledgments

- LEGUS team for the cluster catalog
- Brown & Gnedin (2021) for compiled measurements
- STScI for HST observations
