#!/usr/bin/env python3

import random
import torch
import torch.optim as optim 
import torch.nn as nn
import benchmark_timer
from log_stats import LogStats

#Returns a binary vector that determines whether a particular packet 
#in a sequence will be replaced by the Generator's output.
#Worth noting that we NEVER mask the first value.
#0 means replace, 1 means leave alone.
def get_mask_vector(length = None, eta = 0.2):
    if length is None:
        return None
    
    mask = []
    for k in range(0, length):
        if random.random() < eta:
            mask.append(0)
        else:
            mask.append(1)
            
    mask[0] = 1
    return mask

#Returns a Tensor of the same shape as first_sequence and second_sequence
#According the rule where elements Xi of the first sequence is replaced by
#element Yi of the second_sequence if the entry Ki of mask is equal to 0.
#The resulting vector X is returned.
def get_interleaved_sequence_by_mask(first_sequence, second_sequence, mask):
    assert (first_sequence.shape == second_sequence.shape),"Sequences must have the same shape."
    assert (first_sequence.shape[1] == len(mask)),"Sequences and mask must have the same 1st dimension."
    masked_sequence = first_sequence.copy_(first_sequence)
    
    for i in range(0, len(mask)):
        if mask[i] == 0:
            masked_sequence[0,i] = second_sequence[0,i]
            
    return masked_sequence

def extract(v):
    return v.data.storage().tolist()

#well OK let's train the GAN on a single sequence and see what happens
def train(G, D, data_loader, num_epochs):
    epoch_timer = benchmark_timer.BenchmarkTimer()
    num_epochs = num_epochs
    print_interval = 1
    loss_log_interval = 100
    sample_count = 0 #count samples up to loss_log_interval then log instantaneous loss
    print_stats = True
    eta = 0.2 #probability of masking an element of a sequence
    
    discriminator_learning_rate = 1e-3
    generator_learning_rate = 1e-3
    sgd_momentum = 0.9
    
    discriminator_training_steps = 1
    generator_training_steps = 1
    
    discriminator_fake_error, discriminator_real_error = 0.0, 0.0
    generator_error = 0
    
    generator_losses = []
    discriminator_fake_losses = []
    g_stats = LogStats() 
    df_stats = LogStats()
    
    criterion = nn.BCELoss() #right now the output is binary so this makes sense
    discriminator_optimizer = optim.SGD(D.parameters(), lr=discriminator_learning_rate, momentum=sgd_momentum)
    generator_optimizer = optim.SGD(G.parameters(), lr=discriminator_learning_rate, momentum=sgd_momentum)
    print("Training has started...")
    for epoch in range(num_epochs):
        epoch_timer.start()
        for data_sample in data_loader:

            if data_sample.shape[0] != G.batch_size:
                continue #skip these samples if the batch is the wrong size.

            for discriminator_step in range(discriminator_training_steps):
                #G.init_hidden_state()
                #G.zero_grad()
                D.init_hidden_state()
                D.zero_grad()
                
                #Train D on the real samples
                #print("Sequence length: ", data_sample.shape[1] )
                discriminator_decision_r = D(data_sample)
                discriminator_real_error = criterion(discriminator_decision_r, torch.ones(data_sample.shape[0], data_sample.shape[1], 1))
                discriminator_real_error.backward()
                
                #Train D on the fake samples
                D.init_hidden_state()
                
                #get a real example and mask some of them with generator output
                generator_input_sequence = data_sample
                fake_data = G(generator_input_sequence).detach()  # detach to avoid training G on these labels
                
                #TODO FIXME, is the masker behaving properly with the batch training?
                fake_masked_data = get_interleaved_sequence_by_mask(generator_input_sequence, fake_data, get_mask_vector(data_sample.shape[1], eta))
                

                discriminator_decision_f = D(fake_masked_data)
                discriminator_fake_error = criterion(discriminator_decision_f, torch.zeros(data_sample.shape[0], data_sample.shape[1], 1))
                discriminator_fake_error.backward()
                discriminator_optimizer.step() # Only optimizes D's parameters; changes based on stored gradients from backward()
                
                dre = extract(discriminator_real_error)[0]
                dfe = extract(discriminator_fake_error)[0]
                
                df_stats.log_data(dfe)
        
            for generator_step in range(generator_training_steps):
                #Train G on D's response (but DO NOT train D on these labels)
                G.init_hidden_state()
                G.zero_grad()
                D.init_hidden_state()
                D.zero_grad()
                
                generator_input_sequence = data_sample
                fake_data = G(generator_input_sequence)
                fake_masked_data = get_interleaved_sequence_by_mask(generator_input_sequence, fake_data, get_mask_vector(data_sample.shape[1], eta))
                
                discriminator_decision_dg = D(fake_masked_data)
                generator_error = criterion(discriminator_decision_dg, torch.ones(data_sample.shape[0], data_sample.shape[1], 1)) # Train G to pretend it's genuine
            
                generator_error.backward()
                generator_optimizer.step() # Only optimizes G's parameters
                
                ge = extract(generator_error)[0]
                
                g_stats.log_data(ge)

            sample_count += 1
            if sample_count % loss_log_interval == 0:
                discriminator_fake_losses.append(dfe)
                generator_losses.append(ge)
                
        epoch_timer.stop()    
        #calc average loss for every epoch
        g_stats.log_average()
        df_stats.log_average()
            
        if print_stats:
            if epoch % print_interval == 0 or epoch == num_epochs-1:
                print("Epoch: ", epoch)
                print("D Real Error: ", dre)
                print("D Fake Error: ", dfe)
                print("G Error: ", ge)
                
                #print the timing stats
                seconds_per_epoch = epoch_timer.get_elapsed_seconds()
                remaining_seconds = seconds_per_epoch * (num_epochs - epoch)
                print("Remaining time: " + epoch_timer.seconds_to_readable_time(remaining_seconds))
            
    return G, generator_losses, discriminator_fake_losses, g_stats, df_stats