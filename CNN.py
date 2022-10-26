# overfitting

import sys
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from time import time

dn = 50.
nepochs=300

with open("meteo/2019.csv","r") as f: ls=f.readlines()
trainx = torch.Tensor([float(l.split(',')[1])/dn for l in ls]).view(1,-1,1)
trainy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[6:]]).view(1,-1,1)
with open("meteo/2020.csv","r") as f: ls=f.readlines()
testx = torch.Tensor([float(l.split(',')[1])/dn for l in ls]).view(1,-1,1)
testy = torch.Tensor([float(l.split(',')[1])/dn for l in ls[6:]]).view(1,-1,1)


# trainx = 1, seqlen, 1
# trainy = 1, seqlen, 1
trainds = torch.utils.data.TensorDataset(trainx, trainy)
trainloader = torch.utils.data.DataLoader(trainds, batch_size=1, shuffle=False)
testds = torch.utils.data.TensorDataset(testx, testy)
testloader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False)
crit = nn.MSELoss()


def test(mod):
    mod.train(False)
    totloss, nbatch = 0., 0
    for data in testloader:
        inputs, goldy = data
        haty = mod(inputs.view(1, 1, -1)).view(1, -1, 1)
        loss = crit(haty,goldy)
        totloss += loss.item()
        nbatch += 1
    totloss /= float(nbatch)
    mod.train(True)
    return totloss

def train(mod):
    optim = torch.optim.Adam(mod.parameters(), lr=0.001)
    testLossVector = []
    trainLossVector = []
    epochVector = []
    for epoch in range(nepochs):
        testloss = test(mod)
        totloss, nbatch = 0., 0
        for data in trainloader:
            inputs, goldy = data
            optim.zero_grad()
            haty = mod(inputs.view(1, 1, -1)).view(1, -1, 1)
            loss = crit(haty,goldy)
            totloss += loss.item()
            nbatch += 1
            loss.backward()
            optim.step()
        totloss /= float(nbatch)
        testLossVector.append(testloss)
        trainLossVector.append(totloss)
        epochVector.append(epoch)
        print("err",totloss,testloss)
    print("fin",totloss,testloss,file=sys.stderr)
    return (epochVector, trainLossVector, testLossVector)

layer= (torch.nn.Conv1d(1,1,7,1), torch.nn.ReLU())
mod = torch.nn.Sequential(*layer)
start = time()
epochVector,trainLossVector, testLossVector = train(mod)

print('training time', time()-start)
plt.plot(epochVector, trainLossVector, label="Train Loss")
plt.plot(epochVector, testLossVector, label="Test Loss")
plt.legend(loc="upper right")
plt.xlabel('Epochs', fontsize=9)
plt.title("Prévision sur 7 jours avec le modèle CNN")
plt.show()
