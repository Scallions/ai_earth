import tool  
import numpy as np




def test_score():
    y = np.random.rand(50, 24)
    y_hat = np.random.rand(50, 24)
    score = tool.score(y, y_hat)
    print(score)



if __name__ == "__main__":
    ## test tool
    test_score()
    