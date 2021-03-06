"""
some funcs for data
"""
import netCDF4
import torch
import numpy as np
from constants import *

"""
docker path:
train:
|--tcdata
   |--enso_round1_train_20210201.zip
   |--enso_round1_train_20210201
       |--CMIP_label.nc
       |--CMIP_train.nc
       |--readme.txt
       |--SODA_label.nc
       |--SODA_train.nc

test:
|--tcdata
 	|--enso_round1_test_20210201.zip
	|--enso_round1_test_20210201
		|--test_00001_07_06.npy
		|--test_00014_02_01.npy
		...
"""

class MyDataset(torch.utils.data.Dataset):
    """
    fix start 1
    TODO random start
    """
    def __init__(self, data, label, start_random = True):
        self.data = data
        self.label = label
        self.star_random = start_random
        
    def __len__(self):
        if self.star_random:
            return self.data.shape[0]-12
        else:
            return self.data.shape[0]
        
    def __getitem__(self, index):
        if self.star_random:
            return self.data[index:index+12], self.label[index:index+24]
        else:
            return self.data[index], self.label[index]


def dataloader_from(data, label, start_random=True):
    train_len = int(0.8*data.shape[0])
    if start_random:
        train_dataset = MyDataset(data[:train_len,:,:], label[:train_len+24])
        test_dataset = MyDataset(data[train_len:,:,:], label[train_len:])
    else:
        train_dataset = MyDataset(data[:train_len,:,:], label[:train_len], start_random)
        test_dataset = MyDataset(data[train_len:,:,:], label[train_len:], start_random)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                batch_size = 64, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                batch_size=len(test_dataset), shuffle=False, drop_last=False)

    return train_loader, test_loader

def read_data(only_sst=False, in_range=True, start_random=False, mean=True):
    """
    read train data nc => torch.array
    """
    # data = netCDF4.Dataset("tcdata/enso_round1_train_20210201/SODA_train.nc", "r")
    # label = netCDF4.Dataset("tcdata/enso_round1_train_20210201/SODA_label.nc", "r")
    data = netCDF4.Dataset("tcdata/enso_round1_train_20210201/CMIP_train.nc", "r")
    label = netCDF4.Dataset("tcdata/enso_round1_train_20210201/CMIP_label.nc", "r")
    label = np.array(label.variables['nino'][:])
    sst = np.array(data.variables['sst'][:])
    t300 = np.array(data.variables['t300'][:])
    ua = np.array(data.variables['ua'][:])
    va = np.array(data.variables['va'][:])
    if only_sst:
        features = sst[:,:12].reshape(-1,12,1,24,72)
    else:
        features = np.concatenate([
            sst[:,:12].reshape(-1,12,1,24,72),
            t300[:,:12].reshape(-1,12,1,24,72),
            ua[:,:12].reshape(-1,12,1,24,72),
            va[:,:12].reshape(-1,12,1,24,72)],
            axis=2)
    # nan fill 
    features[np.isnan(features)] = -0.0376
    if in_range:
        features = features[:,:,:,10:13,38:49]
    if start_random:
        c, h, w = features.shape[-2:]
        features = features[:,:12,:,:].reshape(-1,c,h,w)
        label = np.concatenate([label[:,:12].reshape(-1)[12:], label[-1,12:].reshape(-1)], axis=0)
    else:
        label = label[:,12:]
    if mean:
        features = features.mean(-1).mean(-1)
    

    return dataloader_from(features, label, start_random)



def read_test_data(only_sst=False, in_range=True, mean=True):
    """
    read test data npy => torch.array
    """
    data = torch.from_numpy(np.load("tcdata/enso_round1_test_20210201/test_0144-01-12.npy")).float()
    label = torch.from_numpy(np.load("tcdata/enso_round1_test_20210201/label/test_0144-01-12.npy")).float()
    data = data.permute(0,3,1,2)
    if only_sst:
        data = data[:,0,:,:]
    if in_range:
        data = data[:,:,10:13,38:49]
    if mean:
        data = data.mean(-1).mean(-1)
    return data.unsqueeze(0), label.unsqueeze(0)

def read_test_data_nolabel(path,only_sst=False, in_range=True, mean=True):
    data = torch.from_numpy(np.load(path)).float()
    if only_sst:
        data = data[:,0,:,:]
    if in_range:
        data = data[:,:,10:13,38:49]
    if mean:
        data = data.mean(-1).mean(-1)
    return data.unsqueeze(0)

def data_in_range():
    """
    data in range
    """
    pass

