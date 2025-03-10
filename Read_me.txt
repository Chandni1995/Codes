# The project is divided into the following parts:

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





