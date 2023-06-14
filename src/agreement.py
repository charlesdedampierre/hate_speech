import typing as t
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.metrics import cohen_kappa_score, accuracy_score


def plot_table(df_percent_agreement, hate_type, overall_mean):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis("off")

    table_data = table(
        ax,
        df_percent_agreement,
        loc="center",
        cellLoc="center",
        colWidths=[0.2, 0.2, 0.2, 0.1],
    )
    table_data.scale(2, 2)
    table_data.auto_set_font_size(False)
    table_data.set_fontsize(14)

    ax.text(
        0.5,
        1.1,
        hate_type,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="center",
    )
    ax.text(0, -0.1, overall_mean, transform=ax.transAxes, fontsize=12, ha="center")

    return fig


def get_kappa_matrix(coders: t.List[int], labels=t.List[str]):
    # Initialize a matrix to store the pairwise kappa values
    kappa_matrix = np.ones((len(coders), len(coders)))

    for i, j in itertools.combinations(range(len(coders)), 2):
        # Get the labels for the current pair of coders
        labels1 = coders[i]
        labels2 = coders[j]

        # Compute the Cohen's kappa value
        if i == j:
            kappa = 1.0
        else:
            kappa = cohen_kappa_score(labels1, labels2)

        # Store the kappa value in the matrix
        kappa_matrix[i, j] = kappa
        kappa_matrix[j, i] = kappa

    df_matrix = pd.DataFrame(kappa_matrix)
    df_matrix.index = labels
    df_matrix.columns = labels

    return df_matrix


def get_mean(matrix):
    matrix = np.array(matrix)

    # Exclude the diagonal elements
    sum_without_diagonal = np.sum(matrix) - np.trace(matrix)
    count_without_diagonal = np.size(matrix) - np.trace(matrix)

    # Calculate the overall mean without the diagonal
    overall_mean = sum_without_diagonal / count_without_diagonal

    return overall_mean


def get_percent_agreement_matrix(coders: t.List[int], labels=t.List[str]):
    # Initialize a matrix to store the pairwise percent agreement values
    pa_matrix = np.ones((len(coders), len(coders)))

    for i, j in itertools.combinations(range(len(coders)), 2):
        # Get the labels for the current pair of coders
        labels1 = coders[i]
        labels2 = coders[j]

        # Compute the percent agreement value
        if i == j:
            pa = 1.0
        else:
            pa = accuracy_score(labels1, labels2)

        # Store the percent agreement value in the matrix
        pa_matrix[i, j] = pa
        pa_matrix[j, i] = pa

    # Convert the matrix to a DataFrame and set the row and column labels
    df_matrix = pd.DataFrame(pa_matrix)
    df_matrix.index = labels
    df_matrix.columns = labels

    return df_matrix
