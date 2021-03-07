from torch.nn.modules.module import T
import tool  
import data
import models 
import train
import informer

import numpy as np
import torch



def test_score():
    y = np.random.rand(50, 24)
    y_hat = np.random.rand(50, 24)
    score = tool.score(y, y_hat)
    # y = torch.from_numpy(y)
    # y_hat = torch.from_numpy(y_hat)
    score1 = tool.evaluate_metrics(y,y_hat)
    print(score, score1)

def test_read_data():
    dataloader, train_loader = data.read_data()
    for i, (y, x) in enumerate(dataloader):
        print(y.shape, x.shape)
        break

    dataloader, train_loader = data.read_data(mean=False)
    for y, x in dataloader:
        print(y.shape, x.shape)
        break

def test_read_test_data():
    datas, label = data.read_test_data()
    print(datas.shape, label.shape)

def test_model():
    model_ = models.build_model()
    datas, label = data.read_test_data()
    label = label.detach().numpy()
    y = model_(datas).detach().numpy()
    # print(y.shape, label.shape)
    print(tool.score(label, y))

def test_informer():
    model = informer.build_model().double()
    dataloader = data.read_data(in_range=True,mean=True, val=False, start_random=True,dataset="CMIP")
    testloader = data.read_data(in_range=True,mean=True, val=False, start_random=True,dataset="SODA")
    train.train(model, dataloader, testloader)

def test_train():
    model_ = models.build_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = data.read_data(mean=False,in_range=False,val=False, start_random=True)
    train_loader = data.read_data(mean=False,in_range=False,dataset="SODA",val=False)
    datas, label = data.read_test_data(mean=False,in_range=False)
    datas = datas.to(device)
    label = label.detach().numpy()
    model_.to(device)
    model_.eval()
    y = model_(datas).cpu().detach().numpy()
    print(tool.score(label, y))
    model_.train()
    train.train(model_, dataloader, train_loader)
    model_.eval()
    y = model_(datas).cpu().detach().numpy()
    print(tool.score(label, y))
    print(np.abs(y-label))
    # print(y.shape, label.shape)

if __name__ == "__main__":
    tool.set_seed()

    ## test tool
    # test_score()


    ## test data
    # test_read_data()
    # test_read_test_data()

    ## test model
    # test_model()

    # test_train()
    test_informer()