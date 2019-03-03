#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path

#Load a CSV from the UNSW-NB15 dataset into a Pandas DataFrame
def load_unsw_nb15_dataset_as_data_frame(file_path = None, features_path = None):
    if file_path is None or features_path is None:
        return None
        
    
    features_df = pd.read_csv(features_path, encoding="latin-1")
    for i in range(len(features_df.columns.values)):
        features_df.columns.values[i] = str(features_df.columns.values[i]).strip().lower()
    
    #lower case all the types
    for i in range(len(features_df)):
        features_df.loc[i, ['type']] = str(features_df['type'][i]).strip().lower()
        features_df.loc[i, ['name']] = str(features_df['name'][i]).strip().lower()

    packet_data_df = pd.read_csv(file_path, encoding="latin-1", names=features_df['name'], header=None)

    return packet_data_df, features_df
    
    
def test_cases():
    #check that the dataframe is loaded correctly per some pre-determined values
    data_path = "UNSW-NB15_1_clean.csv"
    features_path = "UNSW-NB15_features.csv"
    packet_df, features_df = load_unsw_nb15_dataset_as_data_frame(data_path, features_path)
    assert (packet_df is not None), "Couldn't load packet dataset"
    assert (features_df is not None), "Couldn't load features list"

