import torch
import models
import data
import tool

def train(net, train_loader, test_loader):
    """
    train func
    """
    opt = torch.optim.Adam(net.parameters())
    l = torch.nn.MSELoss()
    for epoch in range(50):
        net.train()
        for i, (x, y) in enumerate(train_loader):
            y = y.float()
            x = x.float()
            opt.zero_grad()
            output = net(x)
            # print(y.shape, output.shape)
            loss = l(y, output)
            loss.backward()
            opt.step()
        if epoch % 5 == 0:
            score1 = tool.score(y.detach().numpy(), output.detach().numpy())
            net.eval()
            for x, y in test_loader:
                y = y.float()
                x = x.float()
                y_hat = net(x)
                score = tool.score(y.detach().numpy(), y_hat.detach().numpy())
            print(f"{epoch}:{i}, loss: {loss.item()}, score: {score}, score1: {score1}")
            


if __name__ == "__main__":
    model = models.build_model()
    data_loader, test_loader = data.read_data()
    train(model, data_loader, test_loader)

