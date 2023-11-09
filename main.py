import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_cleaning import clean_data
from plot_outlier import plot_outliers
from summary_stats import summary_statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Read in the Data set and create a data frame - df
df = pd.read_csv('churn_clean.csv')

x_reference, x_analysis, y, one_hot_columns, binary_columns = clean_data(df)

summary_statistics(df, one_hot_columns, binary_columns, x_analysis)

# Create training and testing set, create and train the model. Print out MSE
# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(x_analysis, y, test_size=0.2, random_state=0)
linreg = LinearRegression().fit(X_train, y_train)
y_pred = linreg.predict(X_test)

# Calculate metrics for Linear regression
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Print Metrics
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

# Plot residuals and predicted values
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.show()

# Fitting a logistic regression model
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)

# Making predictions
y_pred = logreg.predict(X_test)

# Evaluating the model for logistic regression
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Retrieve 'dropped columns' and concatenate to df that was used for analysis and save to new CSV
result_df = pd.concat([x_reference, x_analysis], axis=1)
df.to_csv('churn_prepared.csv')
