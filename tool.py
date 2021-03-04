"""
some tool funcs
"""

import numpy as np


def save_model():
    """
    save model
    """
    pass

def load_model():
    """
    load model
    """
    pass


def lose_func():
    """
    loss func
    """
    pass 

def predict(model=None, dataset=None, out_path=None):
    """
    predict and save result
    """
    pass

def score(y, y_hat):
    """calc score
    y: n x 24
    """
    a = np.zeros((1,24))
    a[0, :4] = 1.5
    a[0, 4:11] = 2
    a[0, 11:18] = 3
    a[0, 18:] = 4
    i = np.arange(1,25,1)
    accskill = (a * np.log(i) * cor(y, y_hat)).sum()
    return 2.0/3 * accskill - rmse(y, y_hat)

def rmse(y, y_hat):
    """calc rmse
    """
    d_y = y - y_hat
    n = y.shape[0]
    return (np.sqrt((d_y**2).sum(0)) / n).sum()

def cor(y, y_hat):
    """calc cor index
    """
    y_mean = y.mean(0)
    y_hat_mean = y_hat.mean(0)
    d_y = y - y_mean 
    d_y_hat = y_hat - y_hat_mean
    cor_up = (d_y * d_y_hat).sum(0)
    cor_down = np.sqrt((d_y**2).sum(0) * (d_y_hat**2).sum(0))
    return cor_up / cor_down
    