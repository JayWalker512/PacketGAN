#!/usr/bin/env python3

import random
import torch
import torch.optim as optim 
import torch.nn as nn
import benchmark_timer
import networks
import progress_bar
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
    #TODO FIXME this does not work correctly with a batch dimension, only masks the first batch and not the others.

    assert (first_sequence.shape == second_sequence.shape),"Sequences must have the same shape."
    assert (first_sequence.shape[1] == len(mask)),"Sequences and mask must have the same 1st dimension."
    masked_sequence = first_sequence.copy_(first_sequence)
    
    for i in range(0, len(mask)):
        if mask[i] == 0:
            masked_sequence[0,i] = second_sequence[0,i]
            
    return masked_sequence

def extract(v):
    return v.data.storage().tolist()

#Currently not used, don't seem to work as desired...
def GeneratorLoss(inputs, targets):
    return -torch.log(inputs.sum()) #possibly better?
    #return torch.log(1-inputs).sum() #inputs should be the discriminators decision

def DiscriminatorLoss(dr_decision, df_decision):
    #we are asked to ascend this gradient, hence the negative to make it descent
    #in this function the loss is combined, so I guess we only need to backwards once?
    return -(torch.log(dr_decision) + torch.log(1 - df_decision)).sum()

#well OK let's train the GAN on a single sequence and see what happens
def train_gan(G, D, data_loader, num_epochs):
    data_timer = benchmark_timer.BenchmarkTimer()
    num_epochs = num_epochs
    print_interval = 1
    loss_log_interval = 100
    sample_count = 0 #count samples up to loss_log_interval then log instantaneous loss
    print_stats = True
    eta = 0.2 #probability of masking an element of a sequence
    
    discriminator_learning_rate = 1e-3
    generator_learning_rate = 1e-3
    sgd_momentum = 0.9
    
    discriminator_training_steps = 5
    generator_training_steps = 1
    
    discriminator_fake_error, discriminator_real_error = 0.0, 0.0
    generator_error = 0
    
    generator_losses = []
    discriminator_fake_losses = []
    g_stats = LogStats() 
    df_stats = LogStats()
    
    criterion = nn.BCELoss() 
    #d_criterion = DiscriminatorLoss
    #g_criterion = GeneratorLoss
    #g_criterion = log(1-D(G(z))) #default loss function in GAN paper but suffers from vanishing gradient
    #though paper says to maximize log(D(G(z))) which would be same as min 1/log(D(G(z)))

    discriminator_optimizer = optim.SGD(D.parameters(), lr=discriminator_learning_rate, momentum=sgd_momentum)
    generator_optimizer = optim.SGD(G.parameters(), lr=discriminator_learning_rate, momentum=sgd_momentum)
    print("Training has started...")
    progress = progress_bar.ProgressBar(total_elements=num_epochs * len(data_loader.dataset), bar_length=30, title="Training")
    for epoch in range(num_epochs):
        for data_sample in data_loader:
            data_timer.start()
            if data_sample.shape[0] != G.batch_size:
                continue #skip these samples if the batch is the wrong size.

            for discriminator_step in range(discriminator_training_steps):
                #D.init_hidden_state()
                #D.zero_grad()
                
                #Train D on the real samples
                #discriminator_decision_r = D(data_sample)
                #discriminator_real_error = d_criterion(discriminator_decision_r, torch.ones(data_sample.shape[0], data_sample.shape[1], 1))
                #discriminator_real_error.backward()
                #discriminator_optimizer.step()
                
                #Train D on the fake samples
                #D.init_hidden_state()
                
                #get a real example and mask some of them with generator output
                #generator_input_sequence = data_sample
                #with torch.no_grad():
                #    fake_data = G(generator_input_sequence)#.detach()  # detach to avoid training G on these labels
                
                #TODO FIXME, is the masker behaving properly with the batch training?
                #fake_masked_data = get_interleaved_sequence_by_mask(generator_input_sequence, fake_data, get_mask_vector(data_sample.shape[1], eta))
                #fake_masked_data = fake_data

                #discriminator_decision_f = D(fake_masked_data)
                #discriminator_fake_error = d_criterion(discriminator_decision_f, torch.zeros(data_sample.shape[0], data_sample.shape[1], 1))
                #discriminator_fake_error.backward()
                #discriminator_optimizer.step() # Only optimizes D's parameters; changes based on stored gradients from backward()
                
                #dre = extract(discriminator_real_error)[0]
                #dfe = extract(discriminator_fake_error)[0]
                
                #df_stats.log_data(dfe)
        
                D.init_hidden_state()
                D.zero_grad()

                #get discriminator real decision
                discriminator_decision_r = D(data_sample)

                with torch.no_grad():
                    fake_data = G(data_sample)#.detach()  # detach to avoid training G on these labels
                
                #get discriminator fake decision
                D.init_hidden_state()
                discriminator_decision_f = D(fake_data)

                discriminator_real_error = criterion(discriminator_decision_r, torch.ones(data_sample.shape[0], data_sample.shape[1], 1))
                discriminator_fake_error = criterion(discriminator_decision_f, torch.zeros(data_sample.shape[0], data_sample.shape[1], 1))
                discriminator_error = (discriminator_real_error + discriminator_fake_error) / 2
                discriminator_error.backward()
                discriminator_optimizer.step()

                dre = extract(discriminator_real_error)[0]
                dfe = extract(discriminator_fake_error)[0]
            
            #This needs to be outside the loop to ensure the discriminator/generator loss lengths match
            discriminator_fake_losses.append(dfe)
            df_stats.log_data(extract(discriminator_error)[0])

            for generator_step in range(generator_training_steps):
                #Train G on D's response (but DO NOT train D on these labels)
                G.init_hidden_state()
                G.zero_grad()
                D.init_hidden_state()
                D.zero_grad()
                
                generator_input_sequence = data_sample
                fake_data = G(generator_input_sequence)
                fake_masked_data = fake_data
                #fake_masked_data = get_interleaved_sequence_by_mask(generator_input_sequence, fake_data, get_mask_vector(data_sample.shape[1], eta))
                
                discriminator_decision_dg = D(fake_masked_data)
                generator_error = criterion(discriminator_decision_dg, torch.ones(data_sample.shape[0], data_sample.shape[1], 1))
                
                generator_error.backward()
                generator_optimizer.step() # Only optimizes G's parameters
                
                ge = extract(generator_error)[0]

            #This needs to be outside the loop to ensure the discriminator/generator loss lengths match
            generator_losses.append(ge)
            g_stats.log_data(ge)




            sample_count += 1
            progress.update(sample_count).draw()
            print("Epoch: ", epoch+1, "/", num_epochs)
            print("D Real Error: ", dre)
            print("D Fake Error: ", dfe)
            print("G Error: ", ge)
            
            #print the timing stats
            data_timer.stop()  
            seconds_per_example = data_timer.get_elapsed_seconds()
            remaining_seconds = seconds_per_example * (progress.total_elements - sample_count) 
            print("Remaining time: " + data_timer.seconds_to_readable_time(remaining_seconds))
                
          
        #calc average loss for every epoch
        g_stats.log_average()
        df_stats.log_average()
            
    return G, D, generator_losses, discriminator_fake_losses, g_stats, df_stats


def test_cases():
    ones_batch = torch.ones(4, 5, 3)
    zeros_batch = torch.zeros(4, 5, 3)
    print(get_interleaved_sequence_by_mask(ones_batch, zeros_batch, get_mask_vector(ones_batch.shape[1], 0.5)))


if __name__ == "__main__":
    test_cases()