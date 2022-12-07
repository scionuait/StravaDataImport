import pandas as pd
import numpy as np


def regression_diagnostic_plot(y_true: numpy.array, y_pred: numpy.array) -> pd.DataFrame:
    """Create diagnostic plots for regression models.

    Parameters
    ----------
    y_true : numpy.array
        target values
    y_pred : numpy.array
        predicted values

    Returns
    -------
    Dataset di valori 

    """
    residuals = y_true - y_pred
    xmin, xmax = y_pred.min(), y_pred.max()
    fig = plt.figure(figsize=(15, 5))
    ax = fig.subplots(nrows=1, ncols=2)
    # Residuals plot
    ax[0].set(title="Residuals vs. fitted plot", xlabel="Fitted values", ylabel="Residuals")
    ax[0].hlines(y=0, xmin=xmin, xmax=xmax, colors="red", linestyles="--", linewidth=2)
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[0])
    # Q-Q plot
    ax[1].set_title("Q-Q plot of residuals")
    qqplot(data=residuals, line="45", fit="True", markersize=5, ax=ax[1])
    plt.tight_layout()
    fig.show()