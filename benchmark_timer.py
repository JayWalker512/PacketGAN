#!/usr/bin/env python3
import time
import datetime

class BenchmarkTimer():
    def __init__(self):
        self.reset()
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.end_time = time.time()
        
    #not really needed as long as you do start and then stop in that order.
    def reset(self):
        self.start_time = 0 
        self.end_time = 0
        
    def get_elapsed_seconds(self):
        return self.end_time - self.start_time
    
    def get_elapsed_readable_time(self):
        return str(datetime.timedelta(seconds=self.get_elapsed_seconds()))
    
    def seconds_to_readable_time(self, num_seconds):
        return str(datetime.timedelta(seconds=num_seconds))
