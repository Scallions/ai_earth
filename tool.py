"""
some tool funcs
"""
import data
from constants import *

import numpy as np
import glob
import os
import zipfile
import random 
import torch

def set_seed(seed = 1):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

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

def predict(model=None, test_data_dir=None, out_path=None):
    """
    predict and save result
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = DEVICE
    model.eval()
    model.to(device)
    for file in glob.iglob(os.path.join(test_data_dir, r"*.npy")):
        print(file)
        if MODEL == "informer":
            model.double()
            s = file.split('_')[-2]
            s = int(s)
            t = np.array([i%12/12.0 - 0.5 for i in range(s,s+38)],dtype=np.float)
            t = np.expand_dims(t,1)
            t = torch.from_numpy(t).double()
            t_n = t[:12].to(device).unsqueeze(0)
            t_p = t[:].to(device).unsqueeze(0)
            x_n = data.read_test_data_nolabel(file, in_range=True, mean=True).double().to(device)
            x_p = torch.zeros(1,26,4).to(device).double()
            x_p = torch.cat([x_n, x_p], dim=1).double().to(device)
            y = model(x_n,t_n,x_p,t_p)[0].reshape(26).cpu().detach().numpy()
            y = np.convolve(y, np.ones((3,))/3, mode="valid")
        else:
            dataset = data.read_test_data_nolabel(file, in_range=False, mean=False).to(device)
            y = model(dataset).reshape(24).cpu().detach().numpy()
        np.save(out_path+file.split("/")[-1], y)

def score(y, y_hat):
    """calc score
    y: n x 24 np.array
    """
    a = np.zeros((24))
    a[:4] = 1.5
    a[4:11] = 2
    a[11:18] = 3
    a[18:] = 4
    i = np.arange(1,25,1)
    accskill = (a * np.log(i) * cor(y, y_hat)).sum()
    return 2.0/3 * accskill - rmse(y, y_hat)

def rmse(y, y_hat):
    """calc rmse
    """
    d_y = y - y_hat
    n = y.shape[0]
    return np.sqrt((d_y**2).sum(0) / n).sum()

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

def make_zip(source_dir='./result/', output_filename = 'result.zip'):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    source_dirs = os.walk(source_dir)
    print(source_dirs)
    for parent, dirnames, filenames in source_dirs:
        print(parent, dirnames)
        for filename in filenames:
            if'.npy'not in filename:
                continue
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)   #????
            zipf.write(pathfile, arcname)
    zipf.close()


def model_convert(model):
    model.eval()
    torch.save(model.state_dict(),"checkpoints/mode-oldversion.pt", _use_new_zipfile_serialization=False)


def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean)**2) * sum((y - y_mean)**2)
    return c1/np.sqrt(c2)
def rmse_(preds, y):
    return np.sqrt(sum((preds - y)**2)/preds.shape[0])
def evaluate_metrics(preds, label):
    # preds = preds.cpu().detach().numpy().squeeze()
    # label = label.cpu().detach().numpy().squeeze()
    preds = preds.squeeze()
    label = label.squeeze()
    acskill = 0
    RMSE = 0
    a = 0
    a = [1.5]*4 + [2]*7 + [3]*7 + [4]*6
    for i in range(24):
        RMSE += rmse_(label[:, i], preds[:, i])
        cor = coreff(label[:, i], preds[:, i])
    
        acskill += a[i] * np.log(i+1) * cor
    return 2/3 * acskill - RMSE