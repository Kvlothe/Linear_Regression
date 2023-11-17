import numpy as np


def summary_statistics(df, one_hot_columns, binary_columns, x_analysis):
    # print out the mode for all the columns separated into one hot columns
    for col in one_hot_columns:
        print(f"Mode of '{col}':")
        print(df[col].mode()[0])
        print()  # Just for an extra blank line for readability

    # print out the count for all the columns separated into one hot columns
    for col in one_hot_columns:
        print(f"Value counts for '{col}':")
        print(df[col].value_counts())
        print()  # Just for an extra blank line for readability

    # print out the mode for all the columns separated into binary columns
    for col in binary_columns:
        print(f"Mode of '{col}':")
        print(df[col].mode()[0])
        print()  # Just for an extra blank line for readability

    # print out the count for all the columns separated into binary columns
    for col in binary_columns:
        print(f"Value counts for '{col}':")
        print(df[col].value_counts())
        print()  # Just for an extra blank line for readability

    # get numeric data for summary statistics
    numeric_data = x_analysis.select_dtypes(include=[np.number])
    numeric_summary_statistics = numeric_data.describe()
    print(df['Bandwidth_GB_Year'].describe())
    print()

    # Split columns into chunks and print them separately
    column_chunks = np.array_split(numeric_data.columns, 10)
    for columns in column_chunks:
        print(numeric_data[columns].describe())
        print()  # Just for an extra blank line for readability

    # create a new CSV file for easier viewing
    numeric_summary_statistics.to_csv('summary_statistics.csv')
    print()  # Just for an extra blank line for readability


def churn_stats(df, churn_col_name):
    # Mode
    print(f"Mode of '{churn_col_name}':")
    print(df[churn_col_name].mode()[0])
    print()  # Just for an extra blank line for readability
    # Value counts
    print(f"Value counts for '{churn_col_name}':")
    print(df[churn_col_name].value_counts())
    print()  # Just for an extra blank line for readability
