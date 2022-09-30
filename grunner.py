from binReader import dataloader
from NNmodel import FCN
import torch
from torch.nn import MSELoss
from torch.optim import Adam

#Hyperparameters
state_dim = 6#speed, weight, frontLength, length, typeCar, index
action_dim = 5
lr = 0.00006
k_epochs = 10
batchSize = 1

fileLocation = "Data/"#input("What is the file location")
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
    return loss

def testLoss(X, Y):
    optim.zero_grad()
    pred = model(X).float()
    loss = lossFunc(pred, Y)
    optim.zero_grad()
    return loss

def createBatch(i, speed, weights, l1, l2, car, data):
    Y = torch.tensor(data[i:i+batchSize], dtype=torch.float16).float()
    return torch.tensor([[speed, weights, l1, l2, car, i+x] for x in range(batchSize)], dtype=torch.float16).float(), Y


def trainEpoch(data):
    lossEpoch = 0.0
    for d in data:
        speed = float(d[1])/80.0
        car = d[2]
        weight, l1, l2 = carInfo[car]
        car = car/3
        weight /= 2500.0
        l1 /= 1.75
        l2 /= 6.0
        for i in range(0, len(d[0])):
            X = torch.tensor([speed, weight, l1, l2, car, (i/len(d[0]))], dtype=torch.float16).float()
            Y = torch.tensor(d[0][i]).float()
            #X, Y = createBatch(i, speed, weight, l1, l2, car, d[0])
            #print(X.shape, Y.shape)
            loss = trainBatch(X, Y)
            lossEpoch += loss


    divider = 0
    for d in data:
        divider += len(d[0])
    return lossEpoch/divider

def testEpoch(data):
    lossEpoch = 0.0
    for d in data:
        speed = float(d[1])/80.0
        car = d[2]
        weight, l1, l2 = carInfo[car]
        car = car/3
        weight /= 2500.0
        l1 /= 1.75
        l2 /= 6.0
        for i in range(0, len(d[0])):
            X = torch.tensor([speed, weight, l1, l2, car, (i/len(d[0]))], dtype=torch.float16).float()
            Y = torch.tensor(d[0][i]).float()
            loss = testLoss(X, Y)
            lossEpoch += loss


    divider = 0
    for d in data:
        divider += len(d[0])
    return lossEpoch/divider

num_epochs = 1000
minLoss = 10000.0
minTLoss = 32000.0
data = [loader.makeData(f) for files in loader.files for f in files]
tester = [data[4], data[23], data[27], data[33]]

del data[4]
del data[23]
del data[27]
del data[33]

for e in range(num_epochs):
    loss = trainEpoch(data)
    tLoss = testEpoch(tester)
    if tLoss < minTLoss:
        minLoss = loss
        minTLoss = tLoss
        torch.save(model.state_dict(), "G_runner/model_loss_{:.5f}.pkl".format(tLoss))
        print("Saved")
    print("Epoch {} gave a training loss of {:.5f} and a testing loss of {}".format(e, loss, tLoss))