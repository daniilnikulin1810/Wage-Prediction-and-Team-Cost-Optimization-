# Wage Prediction and Team Cost Optimization (Prototype)

This project explores the relationship between individual characteristics and wages, and extends this analysis to optimize team composition under uncertainty.

## Status
This project is an early-stage research prototype and will be further developed as part of a Master's thesis.

## Objective
The goal is twofold:
1. Model wage determination using econometric and machine learning methods
2. Optimize team composition while minimizing the risk of high labor costs

## Methods

### Econometric Models
- OLS regression (Mincer equation)
- Quantile regression (q = 0.10, 0.50, 0.90)

### Machine Learning Models
- Random Forest
- XGBoost

### Model Evaluation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### Risk Modeling
- Monte Carlo simulation (100,000 scenarios)
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)

## Key Idea

Instead of only predicting wages, this project focuses on risk-aware decision making:

- Uncertain wages
- Team costs can vary significantly  
- The model simulates many possible scenarios  
- Optimization finds the team with the lowest risk of extreme costs  

## Data
- Source: `wooldridge` dataset (`wage1`)
- Variables used:
  - education (educ)
  - experience (exper, expersq)
  - gender (female)
  - log wage (lwage)

## Output

The model produces:
- Wage predictions (econometric and ML)
- Model comparison (OLS vs RF vs XGBoost)
- Optimal team configuration
- Estimated project risk (CVaR)

## Example Result

- XGBoost typically outperforms OLS and Random Forest
- Risk-aware optimization leads to more stable cost structures
- Final output includes optimal education and experience levels for each team member

## Future Work

- Replace toy dataset with real-world data (EU-SILC)
- Improve model robustness and feature engineering
- Explore nonlinear and stochastic optimization techniques
- Extend to dynamic team allocation problems

## Tech Stack

- Python
- pandas, numpy
- statsmodels
- scikit-learn
- xgboost
- scipy
