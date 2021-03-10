"""
define some constants
such as  data path
...
"""


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

import torch 


MODEL = "informer"
# MODEL = "none"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'