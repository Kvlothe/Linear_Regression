import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.base import clone


def regression(x, y):
    # Splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Initialize and train the Linear Regression model
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)

    # Predict on the testing set
    y_pred = linreg.predict(x_test)

    # Calculate metrics for Linear regression
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the regression equation
    print('Regression equation:')
    print('y = {:.4f}'.format(linreg.intercept_), end='')
    for i, coef in enumerate(linreg.coef_):
        print(' + {:.4f}*{}'.format(coef, x.columns[i]), end='')

    # Print Metrics
    print()
    print()
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared: {r2}')
    print()

    return y_test, y_pred, linreg


def feature_selection(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # Initialize the DecisionTreeClassifier
    dt = DecisionTreeRegressor(random_state=42)

    # Fit the model
    dt.fit(x_train, y_train)

    # Get feature importances
    importances = dt.feature_importances_

    # Transform the importances into a readable format
    feature_importance = zip(x.columns, importances)
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # Print the feature importance
    for feature, importance in feature_importance:
        print(f"Feature: {feature}, Importance: {importance}")

    # Calculate the average feature importance
    average_importance = np.mean(importances)

    # Select features that have importance greater than the average
    selected_features = [feature for feature, importance in feature_importance if importance > average_importance]

    print("Selected features based on importance threshold:")
    print(selected_features)
    print()
    # Create the design matrix X and target vector y using only selected features
    x_selected = x[selected_features]

    return x[selected_features], selected_features, dt


# Feature selection with cross-validation
def feature_selection_with_cv(x, y, estimator, cv=5):
    # Perform cross-validation
    scores = cross_val_score(estimator, x, y, cv=cv, scoring='r2')
    print(f"Cross-validation R^2 scores: {scores}")
    print(f"Average R^2 score: {np.mean(scores)}")

    # Compute feature importances on the full dataset
    # and select features based on the importances from the full model
    estimator_clone = clone(estimator)
    estimator_clone.fit(x, y)
    importances = estimator_clone.feature_importances_

    # Transform the importances into a readable format
    feature_importance = zip(x.columns, importances)
    feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)

    # Print the feature importance
    for feature, importance in feature_importance:
        print(f"Feature: {feature}, Importance: {importance}")

    # Select features based on a more discriminating threshold, like the 75th percentile
    threshold = np.percentile(importances, 75)
    selected_features = [feature for feature, importance in feature_importance if importance > threshold]

    print("Selected features based on importance threshold:")
    for feature in selected_features:
        print(feature)

    # Perform cross-validation again, but only with the selected features
    selected_x = x[selected_features]
    scores_selected = cross_val_score(estimator, selected_x, y, cv=cv, scoring='r2')
    print(f"Cross-validation R^2 scores with selected features: {scores_selected}")
    print(f"Average R^2 score with selected features: {np.mean(scores_selected)}")

    return selected_x, selected_features, estimator_clone


def train_with_regularization(x, y, alpha=1.0):
    # Initialize Ridge regression with an alpha value for regularization strength
    ridge = Ridge(alpha=alpha)

    # Fit the model to the training data
    ridge.fit(x, y)

    # Use cross-validation to evaluate the model
    cv_scores = cross_val_score(ridge, x, y, cv=5)

    # Print the cross-validation scores
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {np.mean(cv_scores)}")

    return ridge
