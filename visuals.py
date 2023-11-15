import seaborn as sns
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt


# Function for plotting all outliers for all integer and float variables
def plot_outliers(df, threshold=0.04, plots_per_page=6, folder_path='Box_Plots'):
    # Selecting numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    # Prepare the directory for saving boxplots
    save_dir = 'Box Plots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # List to keep track of columns with enough outliers to plot
    columns_to_plot = []

    for col in numeric_cols.columns:
        data = numeric_cols[col]

        # Z-Score Method
        z_score = np.abs((data - data.mean()) / data.std())
        outliers_z_score = np.sum(z_score > 3)

        # IQR Method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        outliers_iqr = np.sum((data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr)))

        # Calculate percentage of outliers
        percent_outliers_z_score = outliers_z_score / len(data)
        percent_outliers_iqr = outliers_iqr / len(data)

        # Check if percentage of outliers is above threshold
        if percent_outliers_z_score > threshold or percent_outliers_iqr > threshold:
            columns_to_plot.append(col)

    # Determine how many figures we need
    num_plots = len(columns_to_plot)
    num_figures = math.ceil(num_plots / plots_per_page)

    # Loop through each figure
    for fig_num in range(num_figures):
        fig, axs = plt.subplots(math.ceil(plots_per_page / 2), 2, figsize=(15, 12))
        axs = axs.flatten()  # Flatten the array of axes

        # Loop through each subplot within a figure
        for i in range(plots_per_page):
            plot_number = fig_num * plots_per_page + i

            if plot_number < num_plots:
                col = columns_to_plot[plot_number]
                sns.boxplot(x=df[col], ax=axs[i])
                axs[i].set_title(f"Boxplot of {col}")
                axs[i].set_xlabel(col)

            else:  # Turn off any unused subplots
                axs[i].axis('off')

        plt.tight_layout()
        plt.savefig(f'{folder_path}/Boxplot_Page_{fig_num + 1}.png')
        plt.close()


def heat_map(df, categorical_columns, churn_col, plots_per_page=6, folder_path='Heat_map'):
    num_plots = len(categorical_columns)
    num_pages = math.ceil(num_plots / plots_per_page)

    for page in range(num_pages):
        plt.figure(figsize=(15, 18))  # Adjust the size as needed

        for i in range(plots_per_page):
            plot_number = page * plots_per_page + i

            if plot_number < num_plots:
                col = categorical_columns[plot_number]
                crosstab = pd.crosstab(df[col], df[churn_col])

                plt.subplot(math.ceil(plots_per_page / 2), 2, i + 1)
                sns.heatmap(crosstab, annot=True, fmt='d', cmap='viridis')
                plt.title(f'Heatmap of {col} vs {churn_col}_{page + 1}')
                plt.ylabel(col)
                plt.xlabel(churn_col)

        plt.tight_layout()
        plt.savefig(f'{folder_path}/Heat_map{col}_vs_{churn_col}.png')
        plt.close()


def violin_plot(df, continuous_columns, churn_col, plots_per_page=6, folder_path='Violin_plots'):
    num_plots = len(continuous_columns)
    num_pages = math.ceil(num_plots / plots_per_page)

    for page in range(num_pages):
        plt.figure(figsize=(15, 18))  # Adjust the size as needed

        for i in range(plots_per_page):
            plot_number = page * plots_per_page + i

            if plot_number < num_plots:
                col = continuous_columns[plot_number]

                plt.subplot(math.ceil(plots_per_page / 2), 2, i + 1)
                sns.violinplot(x=df[churn_col], y=df[col])
                plt.title(f'Violin Plot of {col} vs {churn_col}')
                plt.xlabel(churn_col)
                plt.ylabel(col)

        plt.tight_layout()
        plt.savefig(f'{folder_path}/Violin_Plot_{col}_vs_{churn_col}_{page + 1}.png')
        plt.close()


def count_plot(df, categorical_columns, plots_per_page=6, folder_path='Count_Plots'):
    # Calculate the number of pages needed to plot all columns
    num_plots = len(categorical_columns)
    num_pages = math.ceil(num_plots / plots_per_page)

    # Loop through pages
    for page in range(num_pages):
        # Start a new figure
        plt.figure(figsize=(15, 12))  # Adjust the figure size as needed

        # Loop through the positions on the page
        for i in range(plots_per_page):
            # Calculate the position of the current plot
            plot_number = page * plots_per_page + i

            # Check if we still have plots to make
            if plot_number < num_plots:
                # Create subplot position
                plt.subplot(math.ceil(plots_per_page / 2), 2, i + 1)

                # Make the countplot
                sns.countplot(x=categorical_columns[plot_number], data=df)
                plt.title(f'Count Plot of {categorical_columns[plot_number]}')
                plt.xticks(rotation=90)  # Rotate x labels for better readability

        plt.tight_layout()  # Adjust subplots to fit in the figure area
        plt.savefig(f'{folder_path}/categorical_countplots_page_{page + 1}.png')
        plt.close()


def density_plot(df, continuous_columns, plots_per_page=6, folder_path='Density_Plots'):
    # Ensure the directory exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Calculate the number of pages needed
    num_plots = len(continuous_columns)
    num_pages = math.ceil(num_plots / plots_per_page)

    # Plotting
    for page in range(num_pages):
        plt.figure(figsize=(15, 12))  # Adjust the size as needed
        for i in range(plots_per_page):
            plot_index = page * plots_per_page + i
            if plot_index < num_plots:
                plt.subplot(math.ceil(plots_per_page / 2), 2, i + 1)
                col = continuous_columns[plot_index]
                sns.kdeplot(data=df, x=col)
                plt.title(f'Density Plot of {col}')
                plt.xlabel(col)
                plt.ylabel('Density')
            else:  # Turn off any unused subplots
                plt.axis('off')

        # Adjust layout, save and show the figure
        plt.tight_layout()
        plt.savefig(f'{folder_path}/Density_Plot_Page_{page + 1}.png')
        plt.close()


def plot_residuals_and_calculate_rse(y_true, y_pred):
    # Calculate residuals
    residuals = y_true - y_pred

    # Plotting the residual plot
    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.savefig(f'Residual plot.png')
    plt.show()

    # Calculate the Residual Standard Error (RSE)
    rse = np.sqrt(np.sum(residuals ** 2) / (len(residuals) - 2))
    print(f'Residual Standard Error (RSE): {rse:.2f}')\


    return rse
