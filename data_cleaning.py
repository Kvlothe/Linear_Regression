import pandas as pd
import numpy as np
from visuals import plot_outliers


def clean_data(df):

    # Get a list of all the columns sorted by data type, for cleaning and analysis - Easier to look at the data this way
    # rather than in an .xlsx file and group columns by their data type
    dtype_groups = df.columns.groupby(df.dtypes)
    # Print out each data type and its columns of each data type
    for dtype, columns in dtype_groups.items():
        print(f"\nData Type: {dtype}")
        for column in columns:
            print(f"- {column}")

    # Print duplicated rows
    print("Duplicates:")
    print(df[df.duplicated()])

    # I ran a check for missing values and what columns had missing values, entered this line of code after running the
    # below code, but left the checking for missing values code there to verify that there are no missing values.
    df['InternetService'].fillna(df['InternetService'].mode()[0], inplace=True)
    # Get count of missing values and what column/s have missing values
    print("Missing Values")
    missing_values_count = df.isna().sum()
    missing_values_count = missing_values_count[missing_values_count > 0]
    print(missing_values_count)

    # Relabel the columns listed as item1...item8 with appropriate questions
    df.rename(columns={'Item1': 'Timely response',
                       'Item2': 'Timely fixes',
                       'Item3': 'Timely replacement',
                       'Item4': 'Reliability',
                       'Item5': 'Options',
                       'Item6': 'Respectful response',
                       'Item7': 'Courteous exchange',
                       'Item8': 'Evidence of active listening'},
              inplace=True)

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
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers_iqr = np.sum((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))  # IQR Rule for Outliers

        # Storing info
        outliers_info_list.append({'Column': col,
                                   'Num_Outliers_ZScore': outliers_z_score,
                                   'Num_Outliers_IQR': outliers_iqr})

    # Creating DataFrame
    outliers_info = pd.DataFrame(outliers_info_list)

    print(outliers_info)
    # Call the function
    plot_outliers(df)

    # Attempt to cap outliers - did not improve model, new approach needed
    # Outliers - Identifying columns to cap
    # outlier_columns = [col for col in numeric_cols.columns if col in outliers_info['Column'].values]
    #
    # for col in outlier_columns:
    #     df[col] = cap_outliers(df[col])

    # Create two separate data frames, one with the dependant variable > y  and
    # one with all the independent variables > x - one that I will manipulate for analysis
    x = df.drop('Bandwidth_GB_Year', axis=1)
    y = df['Bandwidth_GB_Year']

    # Encoding
    # Create a group for columns that I want to keep around but do not want to use for analysis, then create
    columns_to_keep = ['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'Zip', 'Job', 'Population', 'Lat', 'Lng',
                       'City', 'State', 'County', 'Area', 'PaymentMethod', 'TimeZone']
    x_reference = x[columns_to_keep]
    x_analysis = x.drop(columns=columns_to_keep)

    binary_mapping = {'Yes': 1, 'No': 0, 'DSL': 1, 'Fiber Optic': 0, 'True': 1, 'False': 0}

    binary_columns = ['Techie', 'Tablet', 'Multiple', 'OnlineSecurity', 'OnlineBackup', 'Phone',
                      'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                      'Port_modem', 'Churn', 'InternetService']
    one_hot_columns = ['Marital', 'Gender', 'Contract']

    for col in binary_columns:
        x_analysis[col] = x_analysis[col].map(binary_mapping)
    categorical_columns = one_hot_columns + binary_columns
    continuous_columns_list = x_analysis.drop(columns=categorical_columns).columns.tolist()
    x_analysis = pd.get_dummies(x_analysis, columns=one_hot_columns, drop_first=True)
    # After one-hot encoding, ensure all values are binary (1/0)
    x_analysis = x_analysis.applymap(lambda x: 1 if x == True else (0 if x == False else x))
    df_analysis = df.drop(columns=columns_to_keep)
    print()
    x_analysis.to_csv('churn_prepared.csv')

    return x_reference, x_analysis, y, one_hot_columns, binary_columns, categorical_columns, continuous_columns_list, \
        df_analysis


# Attempt to cap outliers - did not improve model, new approach needed
def cap_outliers(series, lower_percentile=5, upper_percentile=95):
    lower_limit = series.quantile(lower_percentile / 100)
    upper_limit = series.quantile(upper_percentile / 100)
    return series.clip(lower=lower_limit, upper=upper_limit)
