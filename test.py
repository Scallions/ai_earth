import tool  
import data

import numpy as np




def test_score():
    y = np.random.rand(50, 24)
    y_hat = np.random.rand(50, 24)
    score = tool.score(y, y_hat)
    print(score)

def test_read_data():
    dataloader = data.read_data()
    for i, (y, x) in enumerate(dataloader):
        print(y.shape, x.shape)


if __name__ == "__main__":
    ## test tool
    test_score()


    ## test data
    test_read_data()