from binReader import dataloader
from NNmodel import FCN
import torch
from torch.nn import MSELoss
from torch.optim import Adam

#Hyperparameters
state_dim = 6#speed, weight, frontLength, length, typeCar, index
action_dim = 5
lr = 0.00003
k_epochs = 10
batchSize = 5

fileLocation = "C:/Users/Martin/Downloads/CrashEstimation/"#input("What is the file location")
model = FCN(state_dim, action_dim, 128).float()
loader = dataloader(fileLocation)
lossFunc = MSELoss()
optim = Adam(model.parameters(), lr)

carInfo = {0:[1080.65, 1.43, 4.79], 1:[773.15, 0.648, 2.643], 2:[2264.9, 1.5, 5.787]}#weight kg, length to passenger cabin, length of vehicle 0 = Yaris, 2 = Silverado, 1 = ADAS  normalisers: 2.5 tons, 1.75m, 6m, 80kmph

def trainBatch(X, Y):

    optim.zero_grad()
    pred = model(X).float()
    #print(pred.shape)

    loss = lossFunc(pred, Y)
    loss.backward()
    optim.step()
    return loss.mean()

def createBatch(i, speed, weights, l1, l2, car, data):
    Y = torch.tensor(data[(i*batchSize):((i+1)*batchSize)], dtype=torch.float16).float()
    return torch.tensor([[speed, weights, l1, l2, car, (i*batchSize)+x] for x in range(batchSize)], dtype=torch.float16).float(), Y


def trainEpoch(data):
    lossEpoch = 0.0
    for d in data:
        speed = float(d[1]) / 80.0
        car = d[2]
        weight, l1, l2 = carInfo[car]
        car = car / 3
        weight /= 2500.0
        l1 /= 1.75
        l2 /= 6.0
        for i in range(0, len(d[0])//batchSize):
            #X = torch.tensor([speed, weight, l1, l2, car, i], dtype=torch.float16).float()
            #Y = torch.tensor(d[0][i]).float()
            X, Y = createBatch(i, speed, weight, l1, l2, car, d[0])
            #print(X.shape, Y.shape)
            loss = trainBatch(X, Y)
            lossEpoch += loss


    divider = 0
    for d in data:
        divider += len(d[0])
    return lossEpoch/(divider/batchSize)

num_epochs = 1000
minLoss = 1.0
data = [loader.makeData(f) for files in loader.files for f in files]
tester = [data[4], data[9], data[27], data[33]]

del data[4]
del data[9]
del data[27]
del data[33]

for e in range(num_epochs):
    loss = trainEpoch(data)
    if loss < minLoss:
        minLoss = loss
        torch.save(model.state_dict(), "batch/model_loss_{:.2f}.pkl".format(loss))
        print("Saved")
    print("Epoch {} gave a loss of {:.7f}".format(e, loss))