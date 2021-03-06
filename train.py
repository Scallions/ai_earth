import torch
import numpy as np 
import os 
import random

import models
import data
import tool



def train(net, train_loader, test_loader):
    """
    train func
    """
    tool.set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=8e-5)
    l = torch.nn.MSELoss() 
    l.to(device)
    for epoch in range(20):
        net.train()
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
            for x, y in test_loader:
                y = y.float().to(device)
                x = x.float().to(device)
                y_hat = net(x)
                score = tool.score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy())
            print(f"{epoch}:{i}, loss: {loss.item()}, score_test: {score}, score_train: {score1}")
            


if __name__ == "__main__":
    model = models.build_model()
    data_loader, test_loader = data.read_data(mean=False, in_range=False)
    train(model, data_loader, test_loader)

