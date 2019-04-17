#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import math

#expects input to be of shape [batch_size, sequence_length, num_features]
class GRUMapping(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(GRUMapping, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_state = torch.zeros(self.hidden_size) 
        self.mapping = nn.GRU(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        #print("Calling forward")
        self.init_hidden_state()
        x, self.hidden_state = self.mapping(x, self.hidden_state)
        return self.hidden_state

    def init_hidden_state(self):
        #print("Init hidden state")
        self.hidden_state = torch.zeros(1, self.batch_size, self.hidden_size)
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


#TODO FIXME why does the hidden size depend on the batch size and the sequence length?
#TODO FIXME document specifically what shape the input should be...
class Generator(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, noise_size, output_size, batch_size, f, noise_input=False):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.noise_size = noise_size
        self.noise_input = noise_input
        self.h0 = self.init_hidden_state()

        self.gru = nn.GRU(input_size + noise_size, self.hidden_size, batch_first=True)
        self.map = nn.Linear(self.hidden_size, input_size)
        self.f = f

    def forward(self, x):
        if self.noise_input == False: 
            noise = torch.randn(self.batch_size, self.sequence_length, self.noise_size)
            x = torch.cat((x, noise), 2)
        else: #If noise_input is True, then the WHOLE input is normally distributed noise
            x = torch.randn(self.batch_size, self.sequence_length, self.noise_size + self.input_size)

        x, self.h0 = self.gru(x, self.h0)
        x = self.map(x)
        return self.f(x)
    
    def init_hidden_state(self):
        self.h0 = torch.zeros(1,self.batch_size,self.hidden_size)
        return self.h0


class Discriminator(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, output_size, batch_size, f):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.h0 = self.init_hidden_state()

        self.gru = nn.GRU(input_size, self.hidden_size, batch_first=True)
        self.map = nn.Linear(self.hidden_size, 1)
        self.f = f

    def forward(self, x):
        x, self.h0 = self.gru(x, self.h0)
        x = self.map(x)
        return self.f(x)
    
    def init_hidden_state(self):
        self.h0 = torch.zeros(1,self.batch_size,self.hidden_size)
        return self.h0
