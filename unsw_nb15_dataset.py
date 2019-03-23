#!/usr/bin/env python3

import torch
import torch.utils.data
import pandas as pd

class UNSW_NB15(torch.utils.data.Dataset):
    def __init__(self, file_paths = [], sequence_length=25, transform=None):
        #TODO have a sequence_overlap=True flag? Does overlap matter?
        self.transform = transform
        self.sequence_length = sequence_length
        self.columns = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
           'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
           'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
           'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime',
           'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
           'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
           'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
           'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat',
           'label']
        self.dtypes = dtypes = {"scrip": "str",
                                "sport": "int32",
                                "dstip": "str",
                                "dsport": "int32",
                                "proto": "str",
                                "state": "str",
                                "dur": "float64",
                                "sbytes": "int32",
                                "dbytes": "int32",
                                "sttl": "int32",
                                "dttl": "int32",
                                "sloss": "int32",
                                "dloss": "int32",
                                "service": "str",
                                "sload": "float64",
                                "dload": "float64",
                                "spkts": "int32",
                                "dpkts": "int32",
                                "swin": "int32",
                                "dwin": "int32",
                                "stcpb": "int32",
                                "dtcpb": "int32", 
                                "smeansz": "int32",
                                "dmeansz": "int32",
                                "trans_depth": "int32",
                                "res_bdy_len": "int32",
                                "sjit": "float64",
                                "djit": "float64",
                                "stime": "int64",
                                "ltime": "int64",
                                "sintpkt": "float64",
                                "dintpkt": "float64",
                                "tcprtt": "float64",
                                "synack": "float64",
                                "ackdat": "float64",
                                "is_sm_ips_ports": "int32",
                                "ct_state_ttl": "int32",
                                "ct_flw_httpd_mthd": "int32",
                                "is_ftp_login": "int32",
                                "is_ftp_cmd": "int32",
                                "ct_ftp_cmd": "int32",
                                "ct_srv_src": "int32",
                                "ct_dst_ltm": "int32", 
                                "ct_src_ltm": "int32",
                                "ct_src_dport_ltm": "int32",
                                "ct_dst_sport_ltm": "int32",
                                "ct_dst_src_ltm": "int32",
                                "attack_cat": "str",
                                "label": "int32"}
        self.categorical_column_values = {"proto":None, "state":None, "service":None, "attack_cat":None}

        self.dataframe = pd.read_csv(file_paths[0], encoding="latin-1", names=self.columns, header=None, dtype=self.dtypes)
        self.dataframe.sort_values(by=['stime']) #sort chronologically upon loading
        
        #TODO load all the unique values of categorical features at the start
        #and make these accessible via a fast function call.
        for key in self.categorical_column_values:
            self.categorical_column_values[key] = self.dataframe[key].unique()


    def __len__(self):
        return len(self.dataframe.index) - self.sequence_length
    
    def __getitem__(self, index):
        #TODO need error checking for out of bounds?
        #TODO return x,y where y is the category of the example
        #since none corresponds to "normal" data
        
        list_of_dicts = []
        for i in range(index,index+self.sequence_length):
            list_of_dicts.append(self.dataframe.loc[index, :].to_dict())
        
        if self.transform is not None:
            return self.transform(self, list_of_dicts)
        
        return list_of_dicts
    
    #get a list of all the unique labels in the dataset
    def get_labels(self):
        return self.dataframe['label'].unique().tolist()
    
    #get a list of all the unique attack categories in the dataset
    def get_attack_categories(self):
        return self.dataframe['attack_cat'].unique().tolist()
    
    def get_list_of_categories(self, column_name):
        pass #TODO

    #limit the dataset to only examples in the specified category
    def use_only_category(self, category_name):
        if category_name not in self.get_attack_categories():
            return False
        
        new_dataframe = self.dataframe[self.dataframe['attack_cat'] == category_name]
        new_dataframe = new_dataframe.reset_index()
        self.dataframe = new_dataframe
        return True
    
    #limit the dataset to only examples with the specified label
    def use_only_label(self, label):
        if label not in self.get_labels():
            return False
        
        new_dataframe = self.dataframe[self.dataframe['label'] == label]
        new_dataframe = new_dataframe.reset_index()
        self.dataframe = new_dataframe
        return True