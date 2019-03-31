#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import math

#TODO FIXME deal with batch dimension correctly
class LatentSpaceMappingRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LatentSpaceMappingRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_state = torch.zeros(self.hidden_size) 
        self.mapping = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x):
        self.init_hidden_state()
        y = torch.zeros(self.hidden_size)
        for s in range(0, x.shape[0]):
            y = torch.cat((x[s], self.hidden_state), 0)
            self.hidden_state = torch.tanh(self.mapping(y))
            
        return self.hidden_state
        
    def init_hidden_state(self):
        self.hidden_state = torch.zeros(self.hidden_size)
        return self.hidden_state

#expects input to be of shape [batch_size, sequence_length, num_features]
class GRUMapping(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUMapping, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.hidden_state = torch.zeros(self.hidden_size) 
        self.mapping = nn.GRU(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        self.init_hidden_state()
        q = torch.zeros(1, x.shape[1], x.shape[2])
        q[0, :, :] = x[:, :]
        x, self.hidden_state = self.mapping(q, self.hidden_state)
        return self.hidden_state[0][0]

    def init_hidden_state(self):
        self.hidden_state = torch.zeros(1, 1, self.hidden_size)
        return self.hidden_state

#TODO FIXME deal with batch dimension correctly
class LogisticRegressionBinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionBinaryClassifier, self).__init__()
        self.input_size = input_size
        self.mapping = nn.Linear(input_size, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.mapping(x))

#TODO FIXME deal with batch dimension correctly
class NeuralClassifier(nn.Module):
    def __init__(self, input_size, n_classes):
        super(NeuralClassifier, self).__init__()
        self.input_size = input_size
        self.mapping1 = nn.Linear(input_size, input_size)
        self.mapping2 = nn.Linear(input_size, n_classes)
        self.f = torch.sigmoid
    
    def forward(self, x):
        x = self.f(self.mapping1(x))
        return self.f(self.mapping2(x))