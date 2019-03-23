#!/usr/bin/env python3

class LogStats():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.data_accum = []
        self.data_averages = []
        
    def log_data(self, data):
        self.data_accum.append(data)
        
    def log_average(self):
        total = 0
        for l in self.data_accum:
            total += l
            
        length = len(self.data_accum)
        self.data_accum = []
        self.data_averages.append(total / length)
    
    def get_averages(self):
        return self.data_averages
         
def test_log_stats():
    log_stats = LogStats()
    log_stats.log_data(1)
    log_stats.log_data(2)
    log_stats.log_data(3)
    log_stats.log_average()
    log_stats.log_data(2)
    log_stats.log_data(3)
    log_stats.log_data(4)
    log_stats.log_average()
    averages = log_stats.get_averages()
    assert averages[0] == 2,"Incorrect average calculated!"
    assert averages[1] == 3,"Incorrect average calculated!"

if __name__ == "__main__":
    test_log_stats()