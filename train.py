#!/usr/bin/env python3

import random
import torch
import torch.optim as optim 
import torch.nn as nn
import numpy as np
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

#Generator loss for WGAN GP
def GeneratorLoss(df_decision):
    return -torch.mean(df_decision) 

#Discriminator/"Critic" loss for WGAN GP
def DiscriminatorLoss(dr_decision, df_decision): # gradient_penalty, lambda_gp=10):
    return -torch.mean(dr_decision) + torch.mean(df_decision) #+ lambda_gp * gradient_penalty

#This code should be updated to PyTorch 1.0 idioms, some of this is deprecated/unnecessary
#Also not clear how to apply it for sequential data...
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype=torch.float32)
    # Get random interpolation between real and fake samples
    #print("alpha type: ", alpha.dtype)
    #print("real_samples type: ", real_samples.dtype)
    #print("fake_samples type: ", fake_samples.dtype)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #print(interpolates)
    d_interpolates = D(interpolates)
    fake = Variable(torch.tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#well OK let's train the GAN on a single sequence and see what happens
def train_gan(G, D, data_loader, num_epochs):
    data_timer = benchmark_timer.BenchmarkTimer()
    num_epochs = num_epochs
    print_interval = 1
    sample_count = 0 #count samples up to loss_log_interval then log instantaneous loss
    print_stats = True
    eta = 0.2 #probability of masking an element of a sequence
    
    discriminator_learning_rate = 5e-5
    generator_learning_rate = 5e-5
    sgd_momentum = 0.9
    
    discriminator_training_steps = 5
    generator_training_steps = 1
    
    discriminator_fake_error, discriminator_real_error = 0.0, 0.0
    generator_error = 0
    
    generator_losses = []
    discriminator_fake_losses = []
    g_stats = LogStats() 
    df_stats = LogStats()

    #discriminator_optimizer = optim.SGD(D.parameters(), lr=discriminator_learning_rate, momentum=sgd_momentum)
    #generator_optimizer = optim.SGD(G.parameters(), lr=discriminator_learning_rate, momentum=sgd_momentum)
    discriminator_optimizer = optim.RMSprop(D.parameters(), lr=discriminator_learning_rate)
    generator_optimizer = optim.RMSprop(G.parameters(), lr=generator_learning_rate)
    print("Training has started...")
    progress = progress_bar.ProgressBar(total_elements=num_epochs * len(data_loader), bar_length=30, title="Training")
    for epoch in range(num_epochs):
        for data_sample in data_loader:
            data_timer.start()
            if data_sample.shape[0] != G.batch_size:
                continue #skip these samples if the batch is the wrong size.

            for discriminator_step in range(discriminator_training_steps):
                D.init_hidden_state()
                D.zero_grad()

                #get discriminator real decision
                discriminator_decision_r = D(data_sample)

                with torch.no_grad():
                    fake_data = G(data_sample) # no_grad to avoid training G on these labels
                
                #get discriminator fake decision
                D.init_hidden_state()
                discriminator_decision_f = D(fake_data)

                #calculate loss
                #gradient_penalty = compute_gradient_penalty(D, data_sample, fake_data)
                d_loss = DiscriminatorLoss(discriminator_decision_r, discriminator_decision_f)
                d_loss.backward()
                discriminator_optimizer.step()

                for p in D.parameters():
                    p.data.clamp_(-0.01, 0.01) #this is the clipping parameter

                dre = extract(d_loss)[0]
                dfe = extract(d_loss)[0]
            
            #This needs to be outside the loop to ensure the discriminator/generator loss lengths match
            discriminator_fake_losses.append(dfe)
            df_stats.log_data(extract(d_loss)[0])

            for generator_step in range(generator_training_steps):
                #Train G on D's response (but DO NOT train D on these labels)
                G.init_hidden_state()
                G.zero_grad()
                D.init_hidden_state()
                D.zero_grad()
                
                fake_data = G(data_sample)
                discriminator_decision_dg = D(fake_data)
                g_loss = GeneratorLoss(discriminator_decision_dg)
                g_loss.backward()
                generator_optimizer.step() # Only optimizes G's parameters
                
                ge = extract(g_loss)[0]

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