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
    
    
#How can we encode these various features, many of which are discrete integers?
#One-hot or Binary encoding seems logical, using Binary coding to keep things compact.

#Returns a list where each element are a 1 or 0, determining the binary encoding of value with
#at least bits number of bits. If the value cannot be encoding with the requested number of bits,
#None will be returned.
def binary_encode(value, bits):
    encoding = []
    while value != 0:
        encoding.append(value % 2)
        value //= 2
        
    if bits < len(encoding):
        return None #couldn't represent with requested number of bits
    
    while len(encoding) < bits:
        encoding.append(0)
    
    encoding.reverse()
    return encoding

#Takes binary integer in the form of a list containing 1's and 0's. 
#Returns the base-10 (integer) representation of the binary value.
def binary_decode(value):
    if len(value) == 0:
        return None
    
    out = 0
    for i in range(0, len(value)):
        if value[i] == 1:
            out += 2**(len(value) - (i+1))
            
    return out

def float_to_binary(value):
    out = []
    for i in range(len(value)):
        if value[i] >= 0.5:
            out.append(1)
        else:
            out.append(0)
            
    return out


def build_input_feature_tensor(packet_data_dict):
    input_features = []
    
    srcip_segments = str(packet_data_dict['srcip']).split('.')
    srcip_bits = []
    for segment in srcip_segments:
        for k in binary_encode(int(segment), 8):
            srcip_bits.append(k)
    
    dstip_segments = str(packet_data_dict['dstip']).split('.')
    dstip_bits = []
    for segment in dstip_segments:
        for k in binary_encode(int(segment), 8):
            dstip_bits.append(k)
            
    sport = binary_encode(int(packet_data_dict['sport']), 16)
    dport = binary_encode(int(packet_data_dict['dsport']), 16)
    
    
    
    #TODO need to encode the rest of the features buuuuuttttt that can come later.
    
    input_features += srcip_bits + dstip_bits + sport + dport
    
    return torch.tensor(input_features, dtype=torch.float64)
        
#Revert a feature tensor to human readable form
#This working correctly is heavily dependent on sizes and locations chosen in 
#build_input_feature_tensor()
def decode_feature_tensor(feature_tensor):
    output_values = {}
    
    srcip_segments = []
    for i in [0,1,2,3]:
        srcip_segments.append(binary_decode(float_to_binary(feature_tensor[i*8:(i*8)+8])))
        
    srcip_string = ".".join([str(k) for k in srcip_segments])
    
    dstip_segments = []
    for i in [4,5,6,7]:
        dstip_segments.append(binary_decode(float_to_binary(feature_tensor[i*8:(i*8)+8])))
        
    dstip_string = ".".join([str(k) for k in dstip_segments])
    
    sport = binary_decode(float_to_binary(feature_tensor[64:64+16]))
    dport = binary_decode(float_to_binary(feature_tensor[64+16:64+16+16]))
    
    output_values['srcip'] = srcip_string
    output_values['dstip'] = dstip_string
    output_values['sport'] = sport
    output_values['dport'] = dport
    
    return output_values

def build_feature_sequence_tensor(packet_data_dict_list):
    sequence_length = len(packet_data_dict_list)
    example_feature_vector = build_input_feature_tensor(packet_data_dict_list[0])
    seq_out = torch.tensor(()).new_zeros([sequence_length, 1, example_feature_vector.shape[0]])
    
    for i in range(0, sequence_length):
        seq_out[i,0,:] = build_input_feature_tensor(packet_data_dict_list[i])
        
    return seq_out

def decode_feature_sequence_tensor(sequence_tensor):
    seq_out = []

    for i in range(0, sequence_tensor.shape[0]):
        seq_out.append(decode_feature_tensor(sequence_tensor[i,0,:]))
        
    return seq_out



def test_cases():
    #check that the dataframe is loaded correctly per some pre-determined values
    data_path = "UNSW-NB15_1_clean.csv"
    features_path = "UNSW-NB15_features.csv"
    packet_df, features_df = load_unsw_nb15_dataset_as_data_frame(data_path, features_path)
    assert (packet_df is not None), "Couldn't load packet dataset"
    assert (features_df is not None), "Couldn't load features list"

