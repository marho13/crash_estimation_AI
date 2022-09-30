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

fileLocation = "Data/"#input("What is the file location")
model = FCN(state_dim, action_dim, 128).float()
loader = dataloader(fileLocation)
lossFunc = MSELoss()
optim = Adam(model.parameters(), lr)
#model.load_state_dict(torch.load("G_runner/model_loss_0.00309.pkl")) #- speed prediction
model.load_state_dict(torch.load("lowest/model_loss_0.00528.pkl")) #- Deformation prediction
#model.load_state_dict(torch.load("model_loss_0.01049.pkl")) # - Acceleration prediction
carInfo = {0:[1080.65, 1.43, 4.79], 1:[773.15, 0.648, 2.643], 2:[2264.9, 1.5, 5.787]}#weight kg, length to passenger cabin, length of vehicle 0 = Yaris, 2 = Silverado, 1 = ADAS  normalisers: 2.5 tons, 1.75m, 6m

def plotter(data, data1, vehicle, speed):
    car = {0:"Toyota", 1:"ADAS", 2:"Silverado"}
    plt.xlabel("Time - intervals of 10^-1 ms")
    plt.ylabel("Normalized Deformation")#"Speed")#deformation
    plt.plot(data, label="real")
    plt.plot(data1, label="predicted")
    plt.savefig('Deformation/{}-{}.eps'.format(car[vehicle], speed), format='eps')
    plt.clf()
    #plt.show()


def plotPredictedAndReal(data):
    for d in data:
        batchLoss = 0.0
        predictions = []
        speed = float(d[1]) / 80.0
        car = d[2]
        weight, l1, l2 = carInfo[car]
        car = car / 3
        weight /= 2500.0
        l1 /= 1.75
        l2 /= 6.0
        for i in range(0, len(d[0])):
            X = torch.tensor([speed, weight, l1, l2, car, (i/len(d[0]))], dtype=torch.float16).float()
            Y = torch.tensor(d[0][i]).float()
            # X, Y = createBatch(i, speed, weight, l1, l2, car, d[0])
            # print(X.shape, Y.shape)
            pred = model(X).float()
            predictions.append(pred.item())

            loss = lossFunc(pred, Y)
            batchLoss += loss

        print("Loss of {}".format(batchLoss/len(d[0])))
        plotter(d[0], predictions, d[2], float(d[1]))

    pass

num_epochs = 1000
minLoss = 1.0

data = [loader.makeData(f) for files in loader.files for f in files]
tester = [data[4], data[23], data[27], data[33]]
#plotter(tester[0][0], data[0][0], 1, 65.0)
plotPredictedAndReal(tester)