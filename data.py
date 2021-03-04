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
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, index):
        return self.data[index,:,:,:], self.label[index,:]


def dataloader_from(data, label):
    ds = MyDataset(data, label)
    dataloader = torch.utils.data.DataLoader(dataset=ds,
                batch_size = 32, shuffle=True, drop_last=True)
    return dataloader

def read_data():
    """
    read train data nc => torch.array
    """
    data = netCDF4.Dataset("data/enso_round1_train_20210201/SODA_train.nc")
    sst = np.array(data.variables['sst'][:])
    label = netCDF4.Dataset("data/enso_round1_train_20210201/SODA_label.nc")
    label = np.array(label.variables['nino'][:])
    return dataloader_from(sst, label)



def read_test_data():
    """
    read test data npy => torch.array
    """

def data_in_range():
    """
    data in range
    """
    pass

