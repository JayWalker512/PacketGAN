#!/usr/bin/env python3
import time, sys
from IPython.display import clear_output

class ProgressBar():
    def __init__(self, total_elements, bar_length=20, title="Progress"):
        self.title = title
        self.bar_length = bar_length
        self.total_elements = total_elements
        self.num_elements_complete = 0
        self.progress = 0.0
        
    def update(self, num_elements_complete):
        self.num_elements_complete = num_elements_complete
        self.progress = num_elements_complete/self.total_elements
        return self #so we can chain this with the draw method
    
    def draw(self):
        if isinstance(self.progress, int):
            self.progress = float(self.progress)
        if not isinstance(self.progress, float):
            self.progress = 0
        if self.progress < 0:
            self.progress = 0
        if self.progress >= 1:
            self.progress = 1
        block = int(round(self.bar_length * self.progress))
        clear_output(wait = True)
        text = "{2}: [{0}] {1:.1f}%".format( "#" * block + "-" * (self.bar_length - block), self.progress * 100, self.title)
        print(text)