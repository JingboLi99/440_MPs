# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, loss_fn, in_size, out_size, lrate):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> h ->  out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """
        super(NeuralNet, self).__init__()
        #loss function: Cross entropy
        self.loss_fn = loss_fn
    
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.Dropout(0.25),
            nn.Softplus(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, stride=1),
            #nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.Softplus(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.Softplus(),
            nn.MaxPool2d(2)
        )
        
        self.con_layer = nn.Sequential(
            nn.Linear(1024, 170),
            nn.Softplus(),
            nn.Linear(170, 20),
            nn.Softplus(),
            nn.Linear(20, out_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr = lrate, weight_decay = 0.001)
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        #resizing the input for 2d conv:
        #print('pre reshape size: ', x.size())
        out = torch.reshape(x, (x.size(0), 3, 31, 31))
        #print('layer 0 out size: ', out.size())
        out = self.layer1(out)
        #print('layer 1 out size: ', out.size())
        out = self.layer2(out)
        #print('layer 2 out size: ', out.size())
        out = self.layer3(out)
        out = torch.flatten(out, 1)
        #print('flattened size: ', out.size())
        out = self.con_layer(out)
        return out

    def step(self, data, targets):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """

        #   zero the parameter gradients
        self.optimizer.zero_grad()
        
        # run training data forward to get output
        outputs = self.forward(data)
        #calculate loss by comparing ouput to target labels
        loss = self.loss_fn(outputs, targets)
        # #calculated l2 regularization:
        # l2_lambda = 0.001
        # l2_norm = sum(p.pow(2.0).sum() for p in self.network.parameters())
        # loss = loss + l2_lambda + l2_norm
        # backwards: calculate gradients and minimize loss function:
        
        loss.backward()
        self.optimizer.step()
        #return the loss for this particular image
        return loss.item()


def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    
    """
    #parameter settings:
    input_size = 2883
    n_classes = 4
    #Hyperparameters
    lrate = 0.0008
    
    #loss function used
    loss_fn = nn.CrossEntropyLoss()
    #Data loading
    data_and_label = get_dataset_from_arrays(train_set, train_labels)
    train_loader = DataLoader(dataset=data_and_label, batch_size = batch_size, shuffle = False)
    dev_loader = DataLoader(dataset=dev_set, batch_size = batch_size, shuffle = False)
    #Network initialization
    """
    - input size is the size of the input tensor
    - output size is how many classes there are: we want to evaluate the max out of the 4 classes
    - output is ALWAYS the number of classes
    """
    
    model = NeuralNet(loss_fn = loss_fn, in_size = input_size, out_size = n_classes, lrate = lrate)
    loss_to_ret = []
    
    #Training phase: train with training data set
    for epoch in range(epochs):
        for data in train_loader:
            loss_to_ret.append(model.step(data['features'], data['labels']))
            
    
    dev_set = F.normalize(dev_set, p=2.0, dim = 1)
    output = model.forward(dev_set)
    _, predictions = torch.max(output.data, 1)
    return loss_to_ret, predictions.detach().cpu().numpy().astype(int), model
    