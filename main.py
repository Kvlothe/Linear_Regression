import pandas as pd
from data_cleaning import clean_data
from visuals import heat_map
from visuals import scatter_plot
from visuals import violin_plot
from visuals import count_plot
from visuals import density_plot
from summary_stats import summary_statistics
from joblib import dump
from linear_reg import regression
from linear_reg import feature_selection
from linear_reg import feature_selection_with_cv
from visuals import plot_residuals_and_calculate_rse
from sklearn.tree import DecisionTreeRegressor
from linear_reg import train_with_regularization
from linear_reg import regression_statsmodels
from linear_reg import stepwise_selection
from linear_reg import remove_multicollinear_vars


# Read in the Data set and create a data frame - df
df = pd.read_csv('churn_clean.csv')
dependent_variable = 'Bandwidth_GB_Year'
# Clean the data
x_reference, x_analysis, y, one_hot_columns, binary_columns, categorical_columns, continuous_columns_list, \
    df_analysis = clean_data(df)
# print(list(x_analysis.columns))
# x_analysis.to_csv('debug.csv')
# print(x_analysis.dtypes)
# print(y.dtype)

# Get summary statistics
summary_statistics(df, one_hot_columns, binary_columns, df_analysis)
# churn_stats(df, 'Churn')  # Logistical regression dependent
bandwidth_stats = df['Bandwidth_GB_Year']

# Plot uni-variate for categorical and continuous
density_plot(df, continuous_columns_list)
count_plot(df, categorical_columns)

# Plot bi-variate for categorical and continuous against the dependant variable
# heat_map(df, categorical_columns, y)
violin_plot(df, categorical_columns, dependent_variable)
scatter_plot(df, dependent_variable, continuous_columns_list)

# check for multicollinearity
# vif_data = pd.DataFrame()
# vif_data["feature"] = x_analysis.columns
#
# # Calculating VIF for each feature
# vif_data["VIF"] = [variance_inflation_factor(x_analysis.values, i) for i in range(len(x_analysis.columns))]

# print(vif_data)

# linear regression on all independent variables
# regression(x_analysis, y)
regression_statsmodels(x_analysis, y)
estimator = DecisionTreeRegressor(random_state=42)
x_reduced = remove_multicollinear_vars(x_analysis)

# ridge_model = train_with_regularization(x_analysis, y, alpha=1.0)


# feature selection
# x_selected, selected_features, dt_model = feature_selection(x_analysis, y)
# x_selected, selected_features, fitted_estimator = feature_selection_with_cv(x_analysis, y, estimator)
x_selected = stepwise_selection(x_reduced, y)
x_selected = x_analysis[x_selected]

# linear regression on selected features identified from feature selection
y_test, y_pred, linreg_model = regression_statsmodels(x_selected, bandwidth_stats)

# Call the function to plot residuals and calculate RSE (assuming it's defined elsewhere)
rse = plot_residuals_and_calculate_rse(y_test, y_pred)

# Save decision tree model
# dump(dt_model, 'decision_tree_model.joblib')

# Retrieve 'dropped columns' and concatenate to df that was used for analysis and save to new CSV
# result_df = pd.concat([x_reference, x_analysis], axis=1)
# x_analysis.to_csv('churn_prepared.csv')
