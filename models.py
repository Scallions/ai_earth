import torch 
import torch.nn as nn


"""
Define network
"""
class Net(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self,x):
        # x: (b:32,t:12,c:2,h:3,w:11)
        pass 

class simpleSpatailTimeNN(nn.Module):
    def __init__(self, n_cnn_layer:int=1, kernals:list=[3], n_lstm_units:int=64):
        super().__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv4 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.max_pool = nn.AdaptiveAvgPool2d((22, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(88, n_lstm_units, 2, bidirectional=True)
        self.linear = nn.Linear(128, 24)
    def forward(self, x):
        sst = x[:,:,0,:,:]
        t300 = x[:,:,1,:,:]
        ua = x[:,:,2,:,:]
        va = x[:,:,3,:,:]
        for conv1 in self.conv1:
            sst = conv1(sst)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        sst = self.max_pool(sst).squeeze(dim=-1)
        t300 = self.max_pool(t300).squeeze(dim=-1)
        ua = self.max_pool(ua).squeeze(dim=-1)
        va = self.max_pool(va).squeeze(dim=-1)
        
        x = torch.cat([sst, t300, ua, va], dim=-1)
        x = self.batch_norm(x)
        x, _ = self.lstm(x)
        x = self.avg_pool(x).squeeze(dim=-2)
        x = self.linear(x)
        return x

def build_model():
    return simpleSpatailTimeNN()
    # model = torch.nn.Sequential()
    # model.add_module('flatten', torch.nn.Flatten())
    # model.add_module('l1',torch.nn.Linear(48, 24))
    # # model.add_module('bn1', torch.nn.BatchNorm1d())
    # model.add_module('ac1',torch.nn.ReLU())
    # model.add_module('dp1', torch.nn.Dropout(0.25))
    # model.add_module('l2',torch.nn.Linear(24, 12))
    # model.add_module('ac2',torch.nn.ReLU())
    # model.add_module('dp2', torch.nn.Dropout(0.25))
    # model.add_module('l3',torch.nn.Linear(12, 12))
    # model.add_module('ac3',torch.nn.ReLU())
    # model.add_module('dp3', torch.nn.Dropout(0.25))
    # model.add_module('l4',torch.nn.Linear(12, 24))
    # model.add_module('ac4',torch.nn.ReLU())
    # model.add_module('dp4', torch.nn.Dropout(0.25))
    # model.add_module('l5', torch.nn.Linear(24,24))
    # return model