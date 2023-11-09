import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# Function for plotting all outliers for all integer and float variables
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
            plt.title(f"Boxplot of {col} (Z-Score Outliers: {percent_outliers_z_score:.2%}, "
                      f"IQR Outliers: {percent_outliers_iqr:.2%})")
            plt.xlabel(col)
            plt.show()
