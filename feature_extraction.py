#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import unsw_nb15_dataset
import math

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

def get_one_hot(target, num_classes):
    one_hot = [0.0 for c in range(num_classes)]
    one_hot[target] = 1.0
    return one_hot

def from_one_hot(one_hot):
    for c in range(0, len(one_hot)):
        if one_hot[c] == 1.0 or one_hot[c] == 1:
            return c

    return None


#TODO FIXME is this only defined in the scope of the file? Not clear!
one_hot_categorical = True

#TODO FIXME I should get this list from the unsw_nb15_dataset object in a reliably-ordered way
numeric_features = ['dur', 'sbytes',
       'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'sload', 'dload',
       'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
       'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime',
       'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat']

def build_input_feature_tensor(unsw_nb15_dataset, packet_data_dict):
    input_features = []
    
    #source ip
    srcip_segments = str(packet_data_dict['srcip']).split('.')
    srcip_bits = []
    for segment in srcip_segments:
        for k in binary_encode(int(segment), 8):
            srcip_bits.append(k)
    
    #destination ip
    dstip_segments = str(packet_data_dict['dstip']).split('.')
    dstip_bits = []
    for segment in dstip_segments:
        for k in binary_encode(int(segment), 8):
            dstip_bits.append(k)
            

    #source port        
    sport = binary_encode(int(packet_data_dict['sport']), 16)

    #destination port
    dsport = binary_encode(int(packet_data_dict['dsport']), 16)
    
    if one_hot_categorical == True:
        #protocol
        proto_category_index = np.where(unsw_nb15_dataset.categorical_column_values['proto'] == packet_data_dict['proto'])[0][0]
        proto = get_one_hot(proto_category_index, len(unsw_nb15_dataset.categorical_column_values['proto']))

        #state
        state_category_index = np.where(unsw_nb15_dataset.categorical_column_values['state'] == packet_data_dict['state'])[0][0]
        state = get_one_hot(state_category_index, len(unsw_nb15_dataset.categorical_column_values['state']))
    #else:
        #TODO binary encode these categorical values
        #protocol
    #    proto_category_index = np.where(unsw_nb15_dataset.categorical_column_values['proto'] == packet_data_dict['proto'])[0][0]
    #    proto = get_one_hot(proto_category_index, len(unsw_nb15_dataset.categorical_column_values['proto']))

        #state
    #    state_category_index = np.where(unsw_nb15_dataset.categorical_column_values['state'] == packet_data_dict['state'])[0][0]
    #    state = get_one_hot(state_category_index, len(unsw_nb15_dataset.categorical_column_values['state']))


    #TODO FIXME encode 'service'

    numeric_features_encoded = []

    for nf in numeric_features:
        numeric_features_encoded.append(packet_data_dict[nf] / unsw_nb15_dataset.maximums[nf])

    
    input_features += srcip_bits + dstip_bits + sport + dsport + proto + state + numeric_features_encoded
    
    return torch.tensor(input_features, dtype=torch.float64)


#Revert a feature tensor to human readable form
#This working correctly is heavily dependent on sizes and locations chosen in 
#build_input_feature_tensor()
def decode_feature_tensor(unsw_nb15_dataset, feature_tensor):
    output_values = {}
    
    #source ip
    srcip_segments = []
    for i in [0,1,2,3]:
        srcip_segments.append(binary_decode(float_to_binary(feature_tensor[i*8:(i*8)+8])))
        
    srcip_string = ".".join([str(k) for k in srcip_segments])
    output_values['srcip'] = srcip_string

    #dest ip
    dstip_segments = []
    for i in [4,5,6,7]:
        dstip_segments.append(binary_decode(float_to_binary(feature_tensor[i*8:(i*8)+8])))
        
    dstip_string = ".".join([str(k) for k in dstip_segments])
    output_values['dstip'] = dstip_string

    #source port
    sport = binary_decode(float_to_binary(feature_tensor[64:64+16]))
    output_values['sport'] = sport

    #dest port
    dsport = binary_decode(float_to_binary(feature_tensor[64+16:64+16+16]))
    output_values['dsport'] = dsport

    if one_hot_categorical == True:
        #protocol
        proto_fuzzy_one_hot = feature_tensor[96:96+len(unsw_nb15_dataset.categorical_column_values['proto'])]
        proto_one_hot_index = torch.argmax(proto_fuzzy_one_hot)
        proto = unsw_nb15_dataset.categorical_column_values['proto'][proto_one_hot_index]
        output_values['proto'] = proto

        #state
        feature_tensor_state_index = 96+len(unsw_nb15_dataset.categorical_column_values['proto'])
        state_fuzzy_one_hot = feature_tensor[feature_tensor_state_index:feature_tensor_state_index+len(unsw_nb15_dataset.categorical_column_values['state'])]
        state_one_hot_index = torch.argmax(state_fuzzy_one_hot)
        state = unsw_nb15_dataset.categorical_column_values['state'][state_one_hot_index]
        output_values['state'] = state
    #else: 
        #TODO binary decode these values
        #protocol
    #    proto_fuzzy_one_hot = feature_tensor[96:96+len(unsw_nb15_dataset.categorical_column_values['proto'])]
    #    proto_one_hot_index = torch.argmax(proto_fuzzy_one_hot)
    #    proto = unsw_nb15_dataset.categorical_column_values['proto'][proto_one_hot_index]
    #    output_values['proto'] = proto

        #state
    #    feature_tensor_state_index = 96+len(unsw_nb15_dataset.categorical_column_values['proto'])
    #    state_fuzzy_one_hot = feature_tensor[feature_tensor_state_index:feature_tensor_state_index+len(unsw_nb15_dataset.categorical_column_values['state'])]
    #    state_one_hot_index = torch.argmax(state_fuzzy_one_hot)
    #    state = unsw_nb15_dataset.categorical_column_values['state'][state_one_hot_index]
    #    output_values['state'] = state


    #decode numeric features:
    numerical_features_index_start = feature_tensor_state_index + len(unsw_nb15_dataset.categorical_column_values['state'])
    current_numeric_index = numerical_features_index_start
    for nf in numeric_features:
        output_values[nf] = feature_tensor[current_numeric_index].item() * unsw_nb15_dataset.maximums[nf]
        if "int" in unsw_nb15_dataset.dtypes[nf]:
            output_values[nf] = round(output_values[nf])
        current_numeric_index += 1

    
    return output_values

def build_feature_sequence_tensor(unsw_nb15_dataset, packet_data_dict_list):
    sequence_length = len(packet_data_dict_list)
    example_feature_vector = build_input_feature_tensor(unsw_nb15_dataset, packet_data_dict_list[0])
    seq_out = torch.tensor(()).new_zeros([sequence_length, example_feature_vector.shape[0]])
    
    for i in range(0, sequence_length):
        seq_out[i,:] = build_input_feature_tensor(unsw_nb15_dataset, packet_data_dict_list[i])
        
    return seq_out

def decode_feature_sequence_tensor(unsw_nb15_dataset, sequence_tensor):
    seq_out = []

    for i in range(0, sequence_tensor.shape[0]):
        seq_out.append(decode_feature_tensor(unsw_nb15_dataset, sequence_tensor[i,:]))
        
    return seq_out



def test_cases():
    #test one-hot function
    assert get_one_hot(2, 5) == [0.0, 0.0, 1.0, 0.0, 0.0]

    #data_set = unsw_nb15_dataset.UNSW_NB15('/home/jaywalker/MachineLearning/PacketGAN/UNSW-NB15_1_clean.csv',
    #                                   sequence_length=1)

    #print("Original data item:")
    #data_item = data_set[99][0]
    #print(data_item)
    #encoded = build_input_feature_tensor(data_set, data_item)
    #decoded = decode_feature_tensor(data_set, encoded)
    #print("Decoded data item:\n ", decoded)
    #for k in decoded:
    #    assert data_item[k] == decoded[k],"Value prior to encoding does not match decoded value."
    #The above assertions are encapsulated below:

    data_set = unsw_nb15_dataset.UNSW_NB15('UNSW_NB15_full_clean.csv',
                                       sequence_length=5)

    #ensure tensor encoding and decoded is working correctly
    data_item = data_set[0]
    encoded_feature_sequence_tensor = build_feature_sequence_tensor(data_set, data_item)
    #print("Encoded feature tensor shape: ", encoded_feature_sequence_tensor.shape)
    #print("Encoded feature sequence tensor: ", encoded_feature_sequence_tensor)
    decoded_feature_sequence_tensor = decode_feature_sequence_tensor(data_set, encoded_feature_sequence_tensor)
    for t in range(0, len(decoded_feature_sequence_tensor)):
        for k in decoded_feature_sequence_tensor[t]:
            if type(data_item[t][k]) in [float, np.float32, np.float64]:
                assert math.isclose(data_item[t][k], decoded_feature_sequence_tensor[t][k], abs_tol=0.01),"Value " + str(data_item[t][k]) + " != " + str(decoded_feature_sequence_tensor[t][k])
            else: #TODO need to check if integers are close enough 
                assert data_item[t][k] == decoded_feature_sequence_tensor[t][k],"Value " + str(data_item[t][k]) + " != " + str(decoded_feature_sequence_tensor[t][k]) + " type: " + str(type(data_item[t][k]))


if __name__ == "__main__":
    test_cases()

