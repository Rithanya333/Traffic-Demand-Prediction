# Traffic Demand Prediction

## Overview

Traffic Demand Prediction is an advanced machine learning solution developed for the Flipkart GRID Hackathon. The project forecasts traffic demand using spatio-temporal, weather, and road network data through extensive feature engineering and a high-performance ensemble learning framework.

The solution combines multiple gradient boosting and tree-based models with optimized ensemble weighting to deliver accurate and robust traffic demand forecasts.

---

## Problem Statement

Predict normalized traffic demand at specific locations and time intervals using historical traffic patterns, geographic information, weather conditions, and road network attributes.

---

## Key Features

- Advanced temporal feature engineering
- Geospatial analysis using Geohash decoding
- Geographic clustering using K-Means
- Fourier-based cyclic time features
- Leakage-safe target encoding
- Demand lag feature generation
- Weather and road network integration
- Multi-model ensemble learning
- Optuna-based ensemble optimization
- Time-series cross-validation

---

## Dataset Features

### Temporal Features
- Hour
- Minute
- Time Slot
- Day of Week
- Day of Year
- Weekend Indicator

### Geospatial Features
- Geohash
- Latitude
- Longitude
- Geographic Clusters

### Road Features
- Road Type
- Number of Lanes
- Large Vehicle Access
- Landmark Presence

### Weather Features
- Temperature
- Weather Conditions

---

## Methodology

### Data Preprocessing
- Missing value imputation
- Outlier handling
- Timestamp parsing
- Data validation

### Feature Engineering
- Temporal decomposition
- Fourier harmonic features
- Interaction features
- Geographic clustering
- Lag-based demand features

### Target Encoding
Fold-safe smoothed target encoding was applied to high-cardinality categorical features to prevent data leakage.

### Model Training
The ensemble consists of:

- LightGBM (Aggressive)
- LightGBM (Base)
- LightGBM (Alternative)
- LightGBM (Light)
- LightGBM (Very Light)
- XGBoost
- CatBoost
- ExtraTrees Regressor
- HistGradientBoosting Regressor

### Ensemble Optimization
- Optuna-based weight optimization
- Multi-seed bagging
- Weighted blending
- Prediction calibration

---

## Performance

| Model | Mean R² Score |
|---------|-------------|
| LightGBM Base | 0.963 |
| LightGBM Aggressive | 0.963 |
| LightGBM Alternative | 0.963 |
| ExtraTrees | 0.633 |
| XGBoost | 0.588 |
| LightGBM Light | 0.540 |
| LightGBM Very Light | 0.530 |
| HistGradientBoosting | 0.445 |
| CatBoost | 0.139 |

---

## Tech Stack

**Programming Language**
- Python

**Libraries**
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- XGBoost
- CatBoost
- Optuna
- Geohash2

**Development Environment**
- Google Colab

---

## Evaluation Strategy

- 5-Fold Time-Series Cross Validation
- Out-of-Fold Evaluation
- R² Score
- Mean Absolute Error (MAE)

---

## Applications

- Traffic Flow Forecasting
- Smart Transportation Systems
- Fleet Optimization
- Dynamic Resource Allocation
- Congestion Management
- Urban Mobility Planning

---

## Authors

**Rithanya Raj**  
**Anjan Mahapatra**

Developed as part of the Flipkart GRID Hackathon.

---

## License

This project is intended for educational and research purposes.
