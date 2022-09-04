from binReader import dataloader
from NNmodel import FCN
import torch
from torch.nn import MSELoss
from torch.optim import Adam
import matplotlib.pyplot as plt


#Hyperparameters
state_dim = 6#speed, weight, frontLength, length, typeCar, index
action_dim = 5
lr = 0.00001
k_epochs = 10
batchSize = 1

fileLocation = input("What is the file location")
model = FCN(state_dim, action_dim, 128).float()
loader = dataloader(fileLocation)
lossFunc = MSELoss()
optim = Adam(model.parameters(), lr)
model.load_state_dict(torch.load("model_loss_219.78192138671875.pkl"))

carInfo = {0:[1080.65, 1.43, 4.79], 1:[773.15, 0.648, 2.643], 2:[2264.9, 1.5, 5.787]}#weight kg, length to passenger cabin, length of vehicle 0 = Yaris, 2 = Silverado, 1 = ADAS  normalisers: 2.5 tons, 1.75m, 6m

def plotter(data, data1):
    plt.plot(data[0], label="real")
    plt.plot(data1, label="predicted")
    plt.show()


def plotPredictedAndReal(data):
    predictions = []
    speed = float(data[1]) / 80.0
    car = data[2] / 3
    weight, l1, l2 = carInfo[car]
    weight /= 2500.0
    l1 /= 1.75
    l2 /= 6.0
    for i in range(0, len(data[0])):
        X = torch.tensor([speed, weight, l1, l2, car, i], dtype=torch.float16).float()
        pred = model(X).float()
        predictions.append(pred.item())

    plotter(data, predictions)
    pass

num_epochs = 10000
minLoss = 999999.9
data = [loader.makeData(f) for f in loader.files]
tester = data[12]
plotPredictedAndReal(tester)