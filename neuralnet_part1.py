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
within this file, neuralnet_learderboard and neuralnet_part2.py -- the unrevised staff files will be used for all other
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
        self.network = nn.Sequential(nn.Linear(in_size, 170), nn.Sigmoid(), nn.Linear(170, out_size)) 
        #Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr = lrate)
        
    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        out = self.network(x)        
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
    lrate = 0.001
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
            
    #Development phase: develop w development set
    # predictions = []
    # for image in dev_loader:
    #     output = model.forward(image)
    #     _, prediction = torch.max(output.data, 1)
    #     print('pred type: ', type(prediction))
    #     predictions.append(prediction)
    
    output = model.forward(dev_set)
    _, predictions = torch.max(output.data, 1)
    return loss_to_ret, predictions.detach().cpu().numpy().astype(int), model
    