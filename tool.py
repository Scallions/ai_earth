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
    y: n x 24 np.array
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
    if y.shape[0] == 1:
        return 0
    y_mean = y.mean(0)
    y_hat_mean = y_hat.mean(0)
    d_y = y - y_mean 
    d_y_hat = y_hat - y_hat_mean
    cor_up = (d_y * d_y_hat).sum(0)
    cor_down = np.sqrt((d_y**2).sum(0) * (d_y_hat**2).sum(0))
    return cor_up / cor_down
    
def trend_of_ts(ts):
    """
    decompose ts to period signal and noise
    """
    length = len(ts)
    x = np.array(list(range(length))).reshape((length,1))
    sinx = np.sin(x*np.pi*2/12)
    cosx = np.cos(x*np.pi*2/12)
    sin2x = np.sin(2*x*np.pi*2/12)
    cos2x = np.cos(2*x*np.pi*2/12)
#     sin3x = np.sin(3*x*np.pi*2/12)
#     cos3x = np.cos(3*x*np.pi*2/12)
    ones = np.ones((length,1))
#     data = np.hstack((ones, x, sinx, cosx))
    data = np.hstack((ones, x, sinx, cosx, sin2x, cos2x))
#     data = np.hstack((ones, x, sinx, cosx, sin2x, cos2x, sin3x, cos3x))
    b = np.dot(np.dot(np.linalg.inv(np.dot(data.transpose(), data)), data.transpose()), ts)
    ts_hat = np.dot(data, b)
    noise = ts - ts_hat 
    return ts_hat, noise