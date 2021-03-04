"""
predict
"""
import tool
import data



if __name__ == "__main__":
    data_path = ""
    model_path = ""
    out_path = ""
    dataset = data.load_dataset(data_path)
    model = tool.load_model(model_path)
    tool.predict(model, dataset, out_path)