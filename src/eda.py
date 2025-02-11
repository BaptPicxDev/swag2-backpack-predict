# Python modules.
from typing import (
    List,
    Text,
)

# Other modules.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Environment
# sns.set_theme("darkgrid")


# Functions.
def distplot(df: pd.DataFrame, variable_name: Text, xlabel=None, ylabel=None, figsize=(19, 10)) -> None:
    """Generate a distribution plot.

    :param df:
    :param variable_name:
    :param xlabel:
    :param ylabel:
    :param figsize:
    """
    plt.figure(figsize=figsize)
    sns.displot(df, x=variable_name, kind="hist", kde=True)
    plt.title(f"Distribution of {variable_name}")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


def correlation_heatmap(df_correlation: pd.DataFrame, xlabel=None, ylabel=None, figsize=(19, 10)) -> None:
    """Generate a heatmap to see correlation between variables.

    :param df_corrlation:
    :param xlabel:
    :param ylabel:
    :param figsize:
    :example:
    >>> corr = (
            df
            .select_dtypes(include=["float64", "int64"])
            .drop(columns=["Price", "id", "index"], errors='ignore')
            .corr()
        )
    >>> correlation_heatmap(df_correlation=df)
    """
    # Filtering id or index.
    df_correlation.drop(columns=["id", "index"], errors='ignore', inplace=True)
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_correlation, 
        xticklabels=df_correlation.columns.values,
        yticklabels=df_correlation.columns.values,
    )
    plt.title('Correlation Matrix', fontsize=16)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


def compare_predictions_and_real_values(df: pd.DataFrame, xlabel=None, ylabel=None, figsize=(19, 10)) -> None:
    """Draw

    :param df_corrlation:
    :param xlabel:
    :param ylabel:
    :param figsize:
    """
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=df,
        x="index",
        y="predictions",
        label="predictions",
    )
    sns.lineplot(
        data=df,
        x="index",
        y="real_values",
        label="real_values",
    )
    plt.title("Real vs Predictions")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


def draw_count_plot_to_study_features(df: pd.DataFrame, xlabel=None, ylabel=None, figsize=(19, 10)) -> None:
    """Draw multiple count plot in order to understand categorical variables.

    :param df:
    :param xlabel:
    :param ylabel:
    :param figsize:
    :note:
    - Takes a long time. -> 9 minutes for 9 variables.
    - By lowering the plt.subplot size it works faster.
    """
    # Set the figure size for the subplots.
    plt.subplots(figsize=figsize)
    # Compute the number of features.
    df.drop(columns=["id", "index"], errors="ignore", inplace=True)
    feature_list = list(df.select_dtypes(include=["object", "bool"]))
    height = 3
    width = 4
    # Loop through the specified columns
    for index, column_name in enumerate(feature_list):
        # Create subplots in a 3x2 grid
        plt.subplot(width, height, index + 1)
        # Create a countplot for the current column
        sns.countplot(data=df, x=column_name)
        # Adjust subplot layout for better presentation
        plt.tight_layout()
    # Display the subplots
    plt.show()


def vizualize_feature_importance(feature_importance: np.ndarray, feature_names: List[Text], xlabel=None, ylabel=None, figsize=(19, 10)) -> None:
    """Vizualize feature importance.
    Output is in percent.

    :param feature_importance:
    :param feature_names:
    :param xlabel:
    :param ylabel:
    :param figsize:
    """
    plt.figure(figsize=figsize)
    sns.barplot(
        x="feature_importance",
        y="feature_names",
        data=pd.DataFrame({
            "feature_importance": abs(100 * feature_importance).tolist()[0],
            "feature_names": feature_names,
        }),
        palette="viridis",
    )
    plt.title("Feature Importance from Linear Regression")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()