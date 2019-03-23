#!/usr/bin/env python3

import torch
import torch.nn as nn

#I shouldn't actually do the training in this notebook, this is mostly a test to see if I've prepared
#the features correctly for input to some RNN network.

#MODELS: Define Generator model and Discriminator model
#For the time being, this will be a one-layer RNN that is the same width as the input feature tensor

#TODO FIXME why does the hidden size depend on the batch size and the sequence length?

class Generator(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, noise_size, output_size, batch_size, f):
        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.noise_size = noise_size
        self.h0 = self.init_hidden_state()

        self.gru = nn.GRU(input_size + noise_size, self.hidden_size, batch_first=True)
        self.map = nn.Linear(self.hidden_size, input_size)
        self.f = f

    def forward(self, x):
        noise = torch.randn(self.batch_size, self.sequence_length, self.noise_size)
        x = torch.cat((x, noise), 2)
        x, self.h0 = self.gru(x, self.h0)
        x = self.map(x)
        return self.f(x)
    
    def init_hidden_state(self):
        #old
        #self.h0 = torch.zeros(1,self.batch_size,self.hidden_size)
        self.h0 = torch.zeros(self.batch_size,1,self.hidden_size)
        return self.h0


class Discriminator(nn.Module):
    def __init__(self, sequence_length, input_size, hidden_size, output_size, batch_size, f):
        super(Discriminator, self).__init__()
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
        self.h0 = torch.zeros(self.batch_size,1,self.hidden_size)
        return self.h0
