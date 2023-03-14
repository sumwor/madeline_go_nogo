# general utility function needed in data analysis
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


def bootstrap(data, dim, dim0, n_sample):
    # Resample the rows of the matrix with replacement
    if len(data)>0:  # if input data is not empty
        bootstrap_indices = np.random.choice(data.shape[dim], size=(n_sample, data.shape[dim]), replace=True)

        # Bootstrap the matrix along the chosen dimension
        bootstrapped_matrix = np.take(data, bootstrap_indices, axis=dim)

        meanBoot = np.nanmean(bootstrapped_matrix,2)
        bootAve = np.nanmean(bootstrapped_matrix, axis=(1, 2))
        bootHigh = np.nanpercentile(meanBoot, 97.5, axis=1)
        bootLow = np.nanpercentile(meanBoot, 2.5, axis=1)

    else:  # return nans
        bootAve = np.full(dim0, np.nan)
        bootLow = np.full(dim0, np.nan)
        bootHigh = np.full(dim0, np.nan)
        # bootstrapped_matrix = np.array([np.nan])

    # bootstrapped_2d = bootstrapped_matrix.reshape(80,-1)
    # need to find a way to output raw bootstrap results
    tempData = {'bootAve': bootAve, 'bootHigh': bootHigh, 'bootLow': bootLow}
    index = np.arange(len(bootAve))
    bootRes = pd.DataFrame(tempData, index)

    return bootRes