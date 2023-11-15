import pandas as pd
from data_cleaning import clean_data
from visuals import heat_map
from visuals import violin_plot
from visuals import count_plot
from visuals import density_plot
from summary_stats import summary_statistics
from summary_stats import churn_stats
from joblib import dump
from linear_reg import regression
from linear_reg import feature_selection
from linear_reg import feature_selection_with_cv
from visuals import plot_residuals_and_calculate_rse
from sklearn.tree import DecisionTreeRegressor
from linear_reg import train_with_regularization


# Read in the Data set and create a data frame - df
df = pd.read_csv('churn_clean.csv')

# Clean the data
x_reference, x_analysis, y, one_hot_columns, binary_columns, categorical_columns, continuous_columns = clean_data(df)

# Get summary statistics
# summary_statistics(df, one_hot_columns, binary_columns, x_analysis)
# churn_stats(df, 'Churn')  # Logistical regression dependent
bandwidth_stats = df['Bandwidth_GB_Year']

# Plot uni-variate for categorical and continuous
# density_plot(df, continuous_columns)
# count_plot(df, categorical_columns)

# Plot bi-variate for categorical and continuous against the dependant variable
# heat_map(df, categorical_columns, y)
# violin_plot(df, continuous_columns, y)

# linear regression on all independent variables
regression(x_analysis, y)
estimator = DecisionTreeRegressor(random_state=42)

ridge_model = train_with_regularization(x_analysis, y, alpha=1.0)


# feature selection
# x_selected, selected_features, dt_model = feature_selection(x_analysis, y)
x_selected, selected_features, fitted_estimator = feature_selection_with_cv(x_analysis, y, estimator)

# linear regression on selected features identified from feature selection
# y_test, y_pred, lin_model = regression(x_selected, 'Bandwidth_GB_Year')
y_test, y_pred, linreg_model = regression(x_selected, bandwidth_stats)

# Call the function to plot residuals and calculate RSE (assuming it's defined elsewhere)
rse = plot_residuals_and_calculate_rse(y_test, y_pred)

# Save decision tree model
# dump(dt_model, 'decision_tree_model.joblib')

# Retrieve 'dropped columns' and concatenate to df that was used for analysis and save to new CSV
result_df = pd.concat([x_reference, x_analysis], axis=1)
df.to_csv('churn_prepared.csv')
