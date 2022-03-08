#!/usr/bin/env python

import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import transforms
import numpy as np
import csv
from netCDF4 import Dataset
import matplotlib.pyplot as plt

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, layersize, resize):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(resize, layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.ReLU(),
            nn.Linear(layersize, 1505)
        )
    """
        nn.Conv1d(1, 2, 3),
            nn.ReLU(),
            nn.Conv1d(2, 4, 3),
            nn.ReLU(),
            nn.Conv1d(4, 8, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
    """


    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

#creates a mask labeling the points of interest for a max inundation map as 1 and the points to ignore as 0
def createmask(maxinundation1d):
    samplemax = maxinundation1d[0]
    maskarray = np.zeros(len(samplemax))
                
    for i in range(len(samplemax)):
        if(samplemax[i]==-1):
            maskarray[i]=0
        else:
            maskarray[i]=1
            
    return maskarray

#uses the mask to cut down the max inundation input array
def maskinput(maxinundation1d, maskarray):
    count = 0
    for i in range(len(maskarray)):
        if(maskarray[i]==1):
            count+=1
    maskedinput = np.zeros((len(maxinundation1d),count))

    for i in range(len(maxinundation1d)):
        count = 0
        for j in range(len(maskarray)):
            if(maskarray[j]==1):
                maskedinput[i,count] = maxinundation1d[i,j]
                count+=1

    return maskedinput

#defining error as the absolute difference in squares that had above 0.1m inundation
def customloss(outputs, targets):
    outputs = outputs.cpu().detach().numpy()[0]
    targets = targets.cpu().detach().numpy()[0]
    loss = 0
    count = 0

    for i in range(len(outputs)):
        if(outputs[i]>0.1 or targets[i]>0.1):
            loss+=abs(outputs[i]-targets[i])
            count+=1

    loss = float(loss/count)
            
    return loss

#downsamples initial earthquake file to a size of your choice
def downsample(inputarray,resize):
    inputsize = len(inputarray[0])
    ratio = float(inputsize/resize)
    outputarray = np.zeros((3000,resize))

    for k in range(3000):
        for i in range(resize):
            index = int(np.rint(float(ratio*i)))
            if(index<resize):
                outputarray[k,i] = inputarray[k,index]

    return outputarray
    
if __name__ == '__main__':

    #parameters
    testnumber = 300
    epochs = 100
    learningrate = 0.05
    resize = 29
    layersize = 3000

    # Set fixed random number seed
    torch.manual_seed(42)

    #prepare the dataset
    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/largeuplift.txt', "r") as csvfile:
        datax = list(csv.reader(csvfile))
    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/maxinundation2.txt', "r") as csvfile:
        datay = list(csv.reader(csvfile))

    #change everything to numpy arrays for ease of use
    x = np.asarray(datax)
    y = np.asarray(datay)
    x = x.astype(np.float)
    y = y.astype(np.float)
    print(np.shape(x))

    #downsample the input earthquake file to a different size
    x = downsample(x,resize)
    x = x.astype(np.float)

    #mask the extra squares that are in the ocean or too high on land
    maskarray = createmask(y)
    y = maskinput(y,maskarray)
    print(np.shape(y))

    xtrain = torch.tensor(x[0:len(x[:,0])-testnumber,:])
    ytrain = torch.tensor(y[0:len(y[:,0])-testnumber,:])
    xtest = torch.tensor(x[len(x[:,0])-testnumber:,:])
    ytest = torch.tensor(y[len(y[:,0])-testnumber:,:])

    train = data_utils.TensorDataset(xtrain, ytrain)
    trainloader = data_utils.DataLoader(train, batch_size=1, shuffle=False)
    xtest = xtest.unsqueeze(1)
    
    # Initialize the MLP
    mlp = MLP(layersize, resize)
    
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    loss_function = nn.L1Loss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=learningrate)

    epochlossarray = np.zeros(epochs)
    
    # Run the training loop
    for epoch in range(0, epochs): 
        
        epochloss = 0
        # Print epoch
        print('Starting epoch '+str(epoch))
    
        # Set current loss value
        current_loss = 0.0
        epochloss = 0
        myloss = 0
    
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
        
            # Get inputs
            inputs, targets = data
            inputs = inputs.unsqueeze(1)
      
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs.float())
            
            # Compute loss
            loss = loss_function(outputs, targets)
            if(epoch==epochs-1):
                myloss += customloss(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            epochloss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0
        if(epoch==epochs-1):
            myloss = float(myloss/3000)
            print("custom loss: "+str(myloss))

        epochloss = float(epochloss/len(list(train)))
        print("epoch loss: "+str(epochloss))
        epochlossarray[epoch] = epochloss
        
    plt.plot(epochlossarray)
    plt.show()
    
    # Process is complete. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Training process has finished.')

    pred = mlp(xtest.float())
    pred = pred.cpu().detach().numpy()
    ytest = ytest.cpu().detach().numpy()
    
    prediction = np.zeros((64,62))
    actual = np.zeros((64,62))
    error = np.zeros((64,62))
    total = np.zeros((testnumber*3,64,62))

    totalavgerror = 0
    totalavgdiff = 0
    totalavgL1error = 0
    totalavgmiss = 0
    totalavgfalse = 0
    totalavgsuccess = 0

    L1sd = np.zeros(3000)
    confinedsd = np.zeros(3000)
    L1sdnum = 0
    confinedsdnum = 0

    for k in range(testnumber):
        print("saving "+str(k)+" out of "+str(testnumber)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        avgerror = 0
        countactual = 0
        countpred = 0
        countsuccess = 0
        countmask = -1
        realisticcount = 0
        L1loss = 0 
        L1count = 0
        for i in range(int(64*62)):
            
            xx = 62-int(i/62)
            yy = i%62

            if(maskarray[i]==1):
                countmask+=1
                #write prediciton, actual, and error to one file
                actual[xx,yy] = ytest[k,countmask]
                prediction[xx,yy] = pred[k,countmask]
                error[xx,yy] = pred[k,countmask] - ytest[k,countmask]

                L1loss+=abs(pred[k,countmask]-ytest[k,countmask])
                L1count+=1
                
                if(pred[k,countmask]>0.1 or ytest[k,countmask]>0.1):
                    avgerror+=abs(pred[k,countmask]-ytest[k,countmask])
                    realisticcount+=1 #only count a square if either the test or prediciton has an appreciable height
                if(pred[k,countmask]>0.1):
                    countpred+=1
                if(ytest[k,countmask]>0.1):
                    countactual+=1
                if(pred[k,countmask]>0.1 and ytest[k,countmask]>0.1):
                    countsuccess+=1 #counting squares that were both predicted and correct
                    
        avgerror=float(avgerror/realisticcount)
        avgdiff = abs(countpred-countactual)
        avgL1error = float(L1loss/L1count)

        L1sd[k] = avgL1error
        confinedsd[k] = avgerror
        
        total[k*3+0] = actual
        total[k*3+1] = prediction
        total[k*3+2] = error

        totalavgmiss += countpred-countsuccess
        totalavgfalse += countactual-countsuccess
        totalavgsuccess += countsuccess
        totalavgerror += avgerror
        totalavgdiff += avgdiff
        totalavgL1error += avgL1error
        
        print("average confined error: "+str(avgerror))
        print("difference in inundated squares: "+str(countpred-countactual))

    L1sdnum = np.std(L1sd)
    confinedsdnum = np.std(confinedsd)

    print(L1sdnum, confinedsdnum, float(totalavgsuccess/testnumber), float(totalavgfalse/testnumber), float(totalavgmiss/testnumber))
        
    print("TOTAL average diff: "+str(float(totalavgdiff/testnumber)))
    print("TOTAL average confined error: "+str(float(totalavgerror/testnumber)))
    print("TOTAL average L1 error: "+str(float(totalavgL1error/testnumber)))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    out_dataset = Dataset("/home/davidgrzan/Tsunami/cascadia/machinelearning/torchout10.nc", 'w', format='NETCDF4')

    out_dataset.createDimension('type', testnumber*3)
    out_dataset.createDimension('latitude', 64)
    out_dataset.createDimension('longitude', 62)
    lats_data   = out_dataset.createVariable('latitude', 'f4', ('latitude',))
    lons_data   = out_dataset.createVariable('longitude', 'f4', ('longitude',))

    #lats_data[:]       = range(0,1,64)
    #lons_data[:]       = range(0,1,62)
    height_data = out_dataset.createVariable('output', 'f4', ('type','latitude','longitude'))
    
    height_data[:,:,:] = total
    
    out_dataset.close()

    
