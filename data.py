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
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return self.data.shape[0]-12
        
    def __getitem__(self, index):
        return self.data[index:index+12,:,:].mean(-1).mean(-1), self.label[index:index+24]


def dataloader_from(data, label):
    train_len = int(0.8*data.shape[0])
    train_dataset = MyDataset(data[:train_len,:,:], label[:train_len+24])
    test_dataset = MyDataset(data[train_len:,:,:], label[train_len:])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                batch_size = 32, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                batch_size=len(test_dataset), shuffle=False, drop_last=False)

    return train_loader, test_loader

def read_data(only_sst=True, in_range=True):
    """
    read train data nc => torch.array
    """
    data = netCDF4.Dataset("data/enso_round1_train_20210201/SODA_train.nc", "r")
    sst = np.array(data.variables['sst'][:])
    t300 = np.array(data.variables['t300'][:])
    ua = np.array(data.variables['ua'][:])
    va = np.array(data.variables['va'][:])
    # if in_range:
    #     sst = sst[:,:12,10:13,38:49].reshape(-1,12,1,3,11)
    #     t300 = t300[:,:12,10:13,38:49].reshape(-1,12,1,3,11)
    #     ua = ua[:,:12,10:13,38:49].reshape(-1,12,1,3,11)
    #     uv = uv[:,:12,10:13,38:49].reshape(-1,12,1,3,11)
    features = np.concatenate([
        sst[:,:12,10:13,38:49].reshape(-1,1,3,11),
        t300[:,:12,10:13,38:49].reshape(-1,1,3,11),
        ua[:,:12,10:13,38:49].reshape(-1,1,3,11),
        va[:,:12,10:13,38:49].reshape(-1,1,3,11)],
        axis=1)
    
    label = netCDF4.Dataset("data/enso_round1_train_20210201/SODA_label.nc", "r")
    label = np.array(label.variables['nino'][:])
    # label = label.reshape(-1,1)
    label = np.concatenate([label[:,:12].reshape(-1)[12:], label[-1,12:].reshape(-1)], axis=0)
    # features.dtype = 'float32'
    # label.dtype = 'float32'
    return dataloader_from(features, label)



def read_test_data():
    """
    read test data npy => torch.array
    """
    data = np.load("data/test/test_0144-01-12.npy").mean(1).mean(1)
    label = np.load("data/test/label/test_0144-01-12.npy")
    return torch.from_numpy(np.expand_dims(data,0)).float(), torch.from_numpy(np.expand_dims(label,0)).float()

def read_test_data_nolabel(path):
    data = np.load(path).mean(1).mean(1)
    return torch.from_numpy(np.expand_dims(data,0)).float()

def data_in_range():
    """
    data in range
    """
    pass

