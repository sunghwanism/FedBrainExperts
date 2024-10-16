import os
import torch

from DataList import dataset_dict
from src.data.FLDataset import FLDataset




def get_client_dataset(use_data_idx, client_num, _mode, verbose=False, get_info=False, PATH='/NFS/Users/moonsh/data/FLData/'):
    """
    use_data_idx: list of int for the index of the dataset from DataList.py
    client_num: int for the number of clients
    """

    assert len(use_data_idx) == client_num, "The number of clients should be equal to the length of use_data_idx"

    client_dataset_dict = {}

    for client_idx, data_idx in enumerate(use_data_idx):
        dataname = [k for k, v in dataset_dict.items() if v == data_idx][0]

        client_dataset = FLDataset(dataname, PATH, verbose=verbose, 
                                   _mode=_mode, get_info=get_info)
        
        client_dataset_dict[client_idx] = client_dataset

    return client_dataset_dict

