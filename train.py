import torch
import numpy as np 
import os 
import random

import models
import informer
import data
import tool
from constants import *



def train(net, train_loader, test_loader=None, epoch=20):
    """
    train func
    """
    tool.set_seed()
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = DEVICE
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=8e-5)
    l = torch.nn.MSELoss() 
    if MODEL == "informer":
        model.double()
    # l = torch.nn.L1Loss()
    l.to(device)
    for epoch in range(epoch):
        net.train()
        if MODEL == "informer":
            for i, (x, t) in enumerate(train_loader):
                x_n = x[:,:12] # (b, 12, 4)
                t_n = t[:,:12] # (b, 12, 1)
                t_p = t # (b, 38, 1)
                x_p = x.clone() # (b, 38, 4)
                x_p[:,12:] = 0
                x = x.to(device)
                x_n = x_n.to(device)
                t_n = t_n.to(device)
                x_p = x_p.to(device)
                t_p = t_p.to(device)
                opt.zero_grad()
                output = net(x_n,t_n, x_p, t_p)
                # print(y.shape, output.shape)
                loss = l(output[0][:,:,0], x[:,12:,0])
                loss.backward()
                opt.step()
                if i % 5 == 0:
                    print(f"{epoch}:{i} : {loss.item()}")
            if epoch % 2 == 0:
                y = x.cpu().detach().numpy()
                y = y[:,12:12+24,0]
                o = output[0].cpu().detach().numpy()
                o = o[:,:24,0]
                score1 = tool.score(y, o)
                net.eval()
                if test_loader is None:
                    print(f"{epoch}:{i}, loss: {loss.item()}, score_train: {score1}")
                    continue
                for i, (x, t) in enumerate(train_loader):
                    x_n = x[:,:12] # (b, 12, 4)
                    t_n = t[:,:12] # (b, 12, 1)
                    t_p = t # (b, 38, 1)
                    x_p = x.clone() # (b, 38, 4)
                    x_p[:,12:] = 0
                    x = x.to(device)
                    x_n = x_n.to(device)
                    t_n = t_n.to(device)
                    x_p = x_p.to(device)
                    t_p = t_p.to(device)
                    opt.zero_grad()
                    output = net(x_n,t_n, x_p, t_p)
                    y = x.cpu().detach().numpy()
                    y = y[:,12:12+24,0]
                    o = output[0].cpu().detach().numpy()
                    o = o[:,:24,0]
                    score = tool.score(y, o)
                print(f"{epoch}:{i}, loss: {loss.item()}, score_test: {score}, score_train: {score1}")

        else:
            for i, (x, y) in enumerate(train_loader):
                y = y.float().to(device)
                x = x.float().to(device)
                opt.zero_grad()
                output = net(x)
                # print(y.shape, output.shape)
                loss = l(y, output)
                loss.backward()
                opt.step()
                # if i % 10 == 0:
                #     print(f"{epoch}:{i}, loss: {loss.item()}")
            if epoch % 5 == 0:
                score1 = tool.score(y.cpu().detach().numpy(), output.cpu().detach().numpy())
                net.eval()
                if test_loader is None:
                    print(f"{epoch}:{i}, loss: {loss.item()}, score_train: {score1}")
                    continue
                for x, y in test_loader:
                    # gui yi hua 
                    y = y.float().to(device)
                    x = x.float().to(device)
                    y_hat = net(x)
                    score = tool.score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
                print(f"{epoch}:{i}, loss: {loss.item()}, score_test: {score}, score_train: {score1}")
        if epoch % 10 == 9: 
            net.eval()  
            print("save net")
            torch.save(net.state_dict(),f"checkpoints/mode-{MODEL}.pt")
            torch.save(net.state_dict(),f"checkpoints/mode-{MODEL}-oldversion.pt", _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    tool.set_seed(4)
    # model = models.build_model()
    model = informer.build_model()
    # data_loader, test_loader = data.read_data(mean=False, in_range=False)
    data_loader = data.read_data(start_random=True,val=False)
    val_loader = data.read_data(start_random=True,dataset="SODA", val=False)
    train(model, data_loader, val_loader, epoch=60)
    train(model, val_loader,epoch=10)

