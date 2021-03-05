"""
predict
"""
import tool
import data
import models 
import train


if __name__ == "__main__":
    # data_path = "/data/enso_round1_train_20210201"
    # test_data_path = "/data/enso_round1_test_20210201/"
    data_path = "tcdata/enso_round1_train_20210201/"
    test_data_path = "tcdata/enso_round1_test_20210201/"
    model_path = ""
    out_path = "result/"
    train_loader, test_loader = data.read_data(data_path)
    model = models.build_model()
    train.train(model, train_loader, test_loader)
    tool.predict(model, test_data_path, out_path) 
    tool.make_zip()