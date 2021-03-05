import torch 



"""
Define network
"""
class Net(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self,x):
        # x: (b:32,t:12,c:2,h:3,w:11)
        pass 


def build_model():
    model = torch.nn.Sequential()
    model.add_module('flatten', torch.nn.Flatten())
    model.add_module('l1',torch.nn.Linear(48, 12))
    model.add_module('ac1',torch.nn.ReLU())
    model.add_module('dp1', torch.nn.Dropout(0.25))
    model.add_module('l2',torch.nn.Linear(12, 12))
    model.add_module('ac2',torch.nn.ReLU())
    model.add_module('dp2', torch.nn.Dropout(0.25))
    model.add_module('l3', torch.nn.Linear(12,24))
    return model