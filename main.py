import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def plot_outliers(df, threshold=0.05):
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    for col in numeric_cols.columns:
        data = numeric_cols[col]

        # Z-Score Method
        z_score = np.abs((data - data.mean()) / data.std())
        outliers_z_score = np.sum(z_score > 3)

        # IQR Method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = np.sum((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))

        # Calculate percentage of outliers
        percent_outliers_z_score = outliers_z_score / len(data)
        percent_outliers_iqr = outliers_iqr / len(data)

        # Check if percentage of outliers is above threshold
        if percent_outliers_z_score > threshold or percent_outliers_iqr > threshold:
            # Plot boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data)
            plt.title(f"Boxplot of {col} (Z-Score Outliers: {percent_outliers_z_score:.2%}, IQR Outliers: {percent_outliers_iqr:.2%})")
            plt.xlabel(col)
            plt.show()

# Read in the Data set and create a data frame - df
df = pd.read_csv('churn_clean.csv')

# Create two separate data frames, one with the dependant variable > y  and
# one with all the independent variables > x - one that I will manipulate for analysis
X = df.drop('Churn', axis=1)
y = df['Churn']

# Get a list of all the columns sorted by data type, for cleaning and analysis - Easier to look at the data this way
# rather than in an .xlsx file
# Group columns by their dtype
dtype_groups = df.columns.groupby(df.dtypes)
# Print out each data type and its columns of each data type
for dtype, columns in dtype_groups.items():
    print(f"\nData Type: {dtype}")
    for column in columns:
        print(f"- {column}")

# I ran a check for missing values and what columns had missing values, entered this line of code after running the
# below code, but left the checking for missing values code there to verify that there are no missing values.
df['InternetService'].fillna(df['InternetService'].mode()[0], inplace=True)
# Get count of missing values and what column/s have missing values
print("Missing Values")
missing_values_count = df.isna().sum()
missing_values_count = missing_values_count[missing_values_count > 0]
print(missing_values_count)

# Outliers
# Selecting numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64'])

# Initializing a list to store outlier info
outliers_info_list = []

for col in numeric_cols.columns:
    data = numeric_cols[col]

    # Z-Score Method
    z_score = np.abs((data - data.mean()) / data.std())
    outliers_z_score = np.sum(z_score > 3)  # Typically, a Z-Score above 3 is considered as an outlier

    # IQR Method
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers_iqr = np.sum((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))  # IQR Rule for Outliers

    # Storing info
    outliers_info_list.append({'Column': col,
                               'Num_Outliers_ZScore': outliers_z_score,
                               'Num_Outliers_IQR': outliers_iqr})

# Creating DataFrame
outliers_info = pd.DataFrame(outliers_info_list)

print(outliers_info)
# Call the function
plot_outliers(df, threshold=0.045)


##########################################
# Create a group for columns that I want to keep around but do not want to use for analysis, then create
columns_to_keep = ['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'Zip', 'Job', 'Population']
X_reference = X[columns_to_keep]
X_analysis = X.drop(columns=columns_to_keep)

binary_mapping = {'Yes': 1, 'No': 0}

binary_columns = ['Techie', 'Tablet', 'Multiple', 'OnlineSecurity', 'OnlineBackup', 'Phone',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Port_modem']

for col in binary_columns:
    X_analysis[col] = X_analysis[col].map(binary_mapping)

y = y.map(binary_mapping)

one_hot_columns = ['City', 'State', 'County', 'Area', 'TimeZone', 'Marital', 'Gender', 'Contract',
                   'InternetService', 'PaymentMethod']


X_analysis = pd.get_dummies(X_analysis, columns=one_hot_columns, drop_first=True)

########################################
# Create training and testing set, create and train the model. Print out MSE
X_train, X_test, y_train, y_test = train_test_split(X_analysis, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# result_df = pd.concat([X_reference, X_analysis], axis=1)
# result_df = pd.concat([X_reference, predictions], axis=1)

