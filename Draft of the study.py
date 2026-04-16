import pandas as pd
import numpy as np
import wooldridge as woo
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import minimize
df = woo.data('wage1')
print('Ready for the Mincer equation')
print(df.info())
formula = 'lwage ~ educ + exper + expersq + female + educ:female'
ols_model = smf.ols(formula, data=df)
ols_results = ols_model.fit(cov_type='HC3')
print(ols_results.summary())

quant_model = smf.quantreg('lwage ~ educ + exper + expersq + female', df)
res_dict = {}
res_dict['q10'] = quant_model.fit(q=0.10).params
res_dict['q50'] = quant_model.fit(q=0.50).params
res_dict['q90'] = quant_model.fit(q=0.90).params
print(f"{'Factor':<15} | {'q=0.10 (Low)':<15} | {'q=0.50 (Median)':<15} | {'q=0.90 (High)':<15}")
print("-" * 70)
for factor in res_dict['q50'].index:
    q10_val = res_dict['q10'][factor]
    q50_val = res_dict['q50'][factor]
    q90_val = res_dict['q90'][factor]
    print(f"{factor:<15} | {q10_val:>14.4f} | {q50_val:>14.4f} | {q90_val:>14.4f}")
dfML = woo.data('wage1')
X = dfML[['educ', 'exper', 'expersq', 'female']]
y = dfML['lwage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"=== PERFORMANCE (Random Forest) ===")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print(f"=== XGBOOST PERFORMANCE ===")
print(f"MAE: {mae_xgb:.4f}")
print(f"RMSE: {rmse_xgb:.4f}")
X_train_ols = sm.add_constant(X_train)
X_test_ols = sm.add_constant(X_test)
ols_model_bench = sm.OLS(y_train, X_train_ols).fit()
y_pred_ols = ols_model_bench.predict(X_test_ols)
mae_ols = mean_absolute_error(y_test, y_pred_ols)
rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
print(f"=== OLS PERFORMANCE ===")
print(f"MAE: {mae_ols:.4f}")
print(f"RMSE: {rmse_ols:.4f}")
results_data = {
    'Model': ['OLS Regression', 'Random Forest', 'XGBoost'],
    'MAE': [mae_ols, mae, mae_xgb],
    'RMSE': [rmse_ols, rmse, rmse_xgb]
}
summary_df = pd.DataFrame(results_data)
base_rmse = summary_df.loc[0, 'RMSE']
summary_df['Efficiency vs OLS (%)'] = ((base_rmse - summary_df['RMSE']) / base_rmse) * 100
print("=== FINAL MODEL COMPARISON TABLE ===")
print(summary_df.to_string(index=False, float_format="%.4f"))

def team_risk_objective(params):
    p = params
    team_features = pd.DataFrame([
        [p[0], p[1], p[1] ** 2, 0],
        [p[2], p[3], p[3] ** 2, 0],
        [p[4], p[5], p[5] ** 2, 1]
    ], columns=['educ', 'exper', 'expersq', 'female'])
    log_wages = xgb_model.predict(team_features)
    n_sim = 100000
    errors = np.random.normal(0, 0.4264, size=(n_sim, len(log_wages)))
    sim_total_costs = np.sum(np.exp(log_wages + errors), axis=1)
    var_95 = np.percentile(sim_total_costs, 95)
    cvar_95 = sim_total_costs[sim_total_costs >= var_95].mean()
    return cvar_95
bnds = [(12, 18), (0, 40), (12, 18), (0, 40), (12, 18), (0, 40)]
initial_guess = [14, 5, 16, 10, 15, 7]
res = minimize(team_risk_objective, initial_guess, bounds=bnds, method='L-BFGS-B')
opt = res.x
print(f"=== OPTIMIZED TEAM CONFIGURATION (XGBoost + CVaR) ===")
print(f"Employee 1: Educ={opt[0]:.1f}, Exper={opt[1]:.1f}")
print(f"Employee 2: Educ={opt[2]:.1f}, Exper={opt[3]:.1f}")
print(f"Employee 3: Educ={opt[4]:.1f}, Exper={opt[5]:.1f}")
print(f"Minimized Project Risk (CVaR): {res.fun:.2f}")


