"""
predict
"""
import tool
import data
import models 
import informer
import train
from constants import *

import torch

if __name__ == "__main__":
    # data_path = "/data/enso_round1_train_20210201"
    # test_data_path = "/data/enso_round1_test_20210201/"
    data_path = "tcdata/enso_round1_train_20210201/"
    test_data_path = "tcdata/enso_round1_test_20210201/"
    model_path = ""
    out_path = "result/"
    tool.set_seed(4)
    # train_loader = data.read_data(in_range=False,mean=False, val=False, start_random=True)
    model = informer.build_model()
    # model = models.build_model()
    model.load_state_dict(torch.load(f"checkpoints/mode-{MODEL}-oldversion.pt"))
    # tool.model_convert(model)
    # train.train(model, train_loader)
    tool.predict(model, test_data_path, out_path) 
    tool.make_zip()