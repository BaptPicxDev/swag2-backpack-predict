# Python modules.
from typing import (
    Text,
)

# Other modules.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def correlation_heatmap(df_corrlation: pd.DataFrame, xlabel=None, ylabel=None, figsize=(19, 10)) -> None:
    """Generate a heatmap to see correlation between variables.

    :param df_corrlation:
    :param xlabel:
    :param ylabel:
    :param figsize:
    """
    # Filtering id or index.
    df_corrlation.drop(columns=['row_num','start_date','end_date','symbol'], errors='ignore', inplace=True)
    plt.figure(figsize=figsize)
    sns.heatmap(
        df_corrlation, 
        xticklabels=df_corrlation.columns.values,
        yticklabels=df_corrlation.columns.values,
    )
    plt.title('Correlation Matrix', fontsize=16)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()


def compare_predictions_and_real_values(predictions: pd.Series, real_values: pd.Series, xlabel=None, ylabel=None, figsize=(19, 10)) -> None:
    """Draw

    :param df_corrlation:
    :param xlabel:
    :param ylabel:
    :param figsize:
    """
    plt.figure(figsize=figsize)
    sns.lineplot(
        predictions,
        label="predictions",
    )
    sns.lineplot(
        real_values,
        label="real_values",
    )
    plt.title("Real vs Predictions")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.show()
