import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
from torch import nn

import wandb

from src.model import resnet
from src.data.DataList import dataset_dict
from src.data.FLDataset import FLDataset


def generate_model(opt):
    assert opt.model in [
        'resnet', # 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(out_dim=opt.out_dim,)

        elif opt.model_depth == 18:
            model = resnet.resnet18(
                out_dim=opt.out_dim,
)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                out_dim=opt.out_dim,
)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                out_dim=opt.out_dim,
)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                out_dim=opt.out_dim,
)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                out_dim=opt.out_dim,
)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                out_dim=opt.out_dim,
)

    return model


def init_wandb(config):
    if not config.nowandb:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb_run = wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config)
        config.wandb_url = wandb.run.get_url()
        
        return wandb_run


def get_client_dataset(config, client_num, _mode, verbose=False, get_info=False, PATH='/NFS/Users/moonsh/data/FLData/'):
    """
    use_data_idx: list of int for the index of the dataset from DataList.py
    client_num: int for the number of clients
    """

    assert len(config.data_idx) == client_num, "The number of clients should be equal to the length of use_data_idx"

    client_dataset_dict = {}

    for client_idx, data_idx in enumerate(config.data_idx):
        dataname = [k for k, v in dataset_dict.items() if v == data_idx][0]

        client_dataset = FLDataset(dataname, PATH, config, verbose=verbose, 
                                   _mode=_mode, get_info=get_info)
        
        client_dataset_dict[client_idx] = client_dataset

    return client_dataset_dict
