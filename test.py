import tool  
import data
import models 
import train

import numpy as np




def test_score():
    y = np.random.rand(50, 24)
    y_hat = np.random.rand(50, 24)
    score = tool.score(y, y_hat)
    print(score)

def test_read_data():
    dataloader, train_loader = data.read_data()
    for i, (y, x) in enumerate(dataloader):
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

def test_train():
    model_ = models.build_model()
    dataloader, train_loader = data.read_data()
    datas, label = data.read_test_data()
    label = label.detach().numpy()
    y = model_(datas).detach().numpy()
    print(tool.score(label, y))
    train.train(model_, dataloader, train_loader)
    y = model_(datas).detach().numpy()
    print(tool.score(label, y))
    # print(y.shape, label.shape)

if __name__ == "__main__":
    ## test tool
    test_score()


    ## test data
    test_read_data()
    test_read_test_data()

    ## test model
    # test_model()

    test_train()