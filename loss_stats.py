#!/usr/bin/env python3

class LossStats():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.loss_accum = []
        self.loss_averages = []
        
    def log_loss(self, loss):
        self.loss_accum.append(loss)
        
    def log_average(self):
        total = 0
        for l in self.loss_accum:
            total += l
            
        length = len(self.loss_accum)
        self.loss_accum = []
        self.loss_averages.append(total / length)
    
    def get_averages(self):
        return self.loss_averages
         
