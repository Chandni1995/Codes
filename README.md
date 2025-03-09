# Program for Bias Correction and Downscaling of CMIP6 GCMs

This repository contains a collection of R scripts designed for bias correction and downscaling of CMIP6 General Circulation Model (GCM) precipitation data using Quantile Delta Mapping (QDM) and the XGBoost framework. The repository is organized into several parts addressing different resolutions and processing steps for the Godavari study region.

## Overview

The project is divided into the following parts:

- **Part1: Generating Parameter Sets at the Coarser Resolution (1 Degree)**  
  - Reads 1° features and IMD observed precipitation data.
  - Extracts and processes data from multiple netCDF files.
  - Applies QDM for bias correction using historical GCM data from various models.
  - Saves parameter sets and bias correction coefficients.

- **Part2: Generating Parameter Sets at the Finer Resolution (0.25 Degree)**  
  - Reads 0.25° features and corresponding IMD observed precipitation data.
  - Processes finer resolution netCDF files and applies QDM.
  - Saves resulting parameter sets for the Godavari region.

- **Part3: Stacking of Parameter Sets at the Coarser Resolution**  
  - Combines individual parameter set files into a single stacked dataset.
  - Applies thresholding to classify raw and bias-corrected precipitation data.

- **Part4: XGBoost Setup and Hyperparameter Optimization**  
  - Implements Bayesian Optimization to tune key hyperparameters for both classification and regression XGBoost models.
  - Optimizes parameters such as learning rate, number of estimators, max depth, min child weight, subsample, colsample by tree, and gamma.
  - Trains the final models using the best-found configurations.

- **Part5: Bias-Correction and Downscaling at the Finer Resolution using XGBoost**  
  - Uses the pre-trained XGBoost models to predict classification (rainy vs. non-rainy days) and regression (quantitative precipitation) on fine resolution data.
  - Saves predictions and computes performance metrics (e.g., NSE).

## Requirements

The R scripts require the following packages:

- hydroGOF
- ncdf4
- qmap
- ncdf4.helpers
- DataCombine
- chron
- xgboost
- caret
- caTools
- e1071
- rBayesianOptimization

You can install these packages using the following command in R:

```r
install.packages(c("hydroGOF", "ncdf4", "qmap", "ncdf4.helpers", "DataCombine", "chron", "xgboost", "caret", "caTools", "e1071", "rBayesianOptimization"))



Setup
Working Directory:
Replace all instances of YOUR_DIRECTORY_PATH in the scripts with the full path to your working directory. This directory should contain your input data (netCDF files, CSV feature files) and desired output directories.

Input Data:

Place the netCDF files for IMD and CMIP6 GCM data in the appropriate subdirectories (e.g., inputs/Godavari/, inputs/IMD_1_degree_rainfall/, inputs/IMD_0.25_degree_rainfall/).
Ensure the CSV files (features_1_degree.csv and features_0.25_degree.csv) with grid features are located in the inputs/ folder.
Usage
The repository contains multiple scripts corresponding to the parts described above. Run the scripts sequentially as follows:

Part1: Generate parameter sets at 1° resolution and apply QDM for bias correction.
Part2: Generate parameter sets at 0.25° resolution and apply QDM.
Part3: Stack the generated parameter sets for the coarser resolution, applying thresholding for classification.
Part4: Set up the XGBoost framework and perform Bayesian hyperparameter optimization.
Part5: Use the optimized XGBoost models for bias correction and downscaling at finer resolution.
Each script is self-contained and includes comments explaining its function and usage. Make sure to update the working directory paths as specified before running the scripts.

Model Training and Evaluation
Classification:
XGBoost is used to classify rainy vs. non-rainy days. Hyperparameter tuning is performed using Bayesian Optimization to select the best parameters from the following ranges:

Learning rate (eta): (0.01, 1)
Number of Estimators (nrounds): (100, 1000)
Maximum Depth: (1, 25)
Minimum Child Weight: (1, 20)
Subsample: (0.3, 1)
Colsample by Tree: (0.3, 1)
Gamma: (0, 1)
Regression:
XGBoost is also used for regression to predict precipitation values. Similar hyperparameter tuning is performed using Bayesian Optimization.

Performance metrics are computed using the hydroGOF package.

Citation
If you use this code in your research, please cite the following works:

Thakur al., 2025.

Contact
For any questions or suggestions, please contact Dr. Chandni at chandnithakur2@gmail.com

