from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Divide X e y in training set e test set.
    """
    n_samples = x.shape[0]
    n_tr = int(np.floor(n_samples * tr_fraction))
    
    Xtr = x[:n_tr]
    ytr = y[:n_tr]
    Xts = x[n_tr:]
    yts = y[n_tr:]
    
    return Xtr, ytr, Xts, yts