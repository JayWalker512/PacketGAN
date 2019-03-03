#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path

def test_fn():
	print("Testing the ability to import functions.")
	
#Load a CSV from the UNSW-NB15 dataset into a Pandas DataFrame
def load_unsw_nb15_dataset_as_data_frame(filePath = None, featuresPath = None):
	if filePath is None or featuresPath is None:
		return None
		
	
