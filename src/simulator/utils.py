import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import torch
from torch import nn

import wandb

from src.model import resnet
from src.model import RepResNet
from src.data.DataList import dataset_dict
from src.data.FLDataset import FLDataset


def generate_model(opt):
    assert opt.model in [
        'resnet', 'RepResNet'
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
                out_dim=opt.out_dim
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
    elif opt.model == 'RepResNet':
        assert opt.model_depth in [10, 18, 34, 50, 101]
        if opt.model_depth == 10:
            model = RepResNet.Represnet10(out_dim=opt.out_dim,)

        elif opt.model_depth == 18:
            model = RepResNet.Represnet18(
                out_dim=opt.out_dim,)
        elif opt.model_depth == 34:
            model = RepResNet.Represnet34(
                out_dim=opt.out_dim,)
        elif opt.model_depth == 50:
            model = RepResNet.Represnet50(
                out_dim=opt.out_dim)
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


def MergeClientDataset(DatasetDict, num_clients):
    merged_dataset = torch.utils.data.ConcatDataset([DatasetDict[i] for i in range(num_clients)])

    return merged_dataset

def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    ages = torch.stack([item[1] for item in batch])
    
    if len(batch[0]) > 2:
        subjects = [item[2] for item in batch]
        sexes = [item[3] for item in batch]
        return images, ages, subjects, sexes
    
    return images, ages

def get_key_by_value(d, value):
    return [key for key, val in d.items() if val == value][0]


def get_activation(activation, model_name, layer_name):
    def hook(module, input, output):
        activation[model_name][layer_name] = output.detach()
    return hook

def register_hooks_for_model(model, model_name, activation):
    hooks = []
    for name, layer in model.named_modules():
        if (isinstance(layer, nn.Conv3d)) and ('downsample' not in name):
            hook = layer.register_forward_hook(get_activation(activation, model_name, name))
            hooks.append(hook)
    return hooks

def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()

def get_activation_for_models(local_model, glob_model, img):
    loc_result = []
    glob_result = []
    target_layer = ['layer1.2.conv2', 'layer2.3.conv2', 'layer3.5.conv2', 'layer4.2.conv2']

    activation = {'Local': {}, 'Global': {}}

    hooks_model_a = register_hooks_for_model(local_model, 'Local', activation)
    hooks_model_b = register_hooks_for_model(glob_model, 'Global', activation)

    _ = local_model(img, [1, 1, 1, 1])
    _ = glob_model(img, [1, 1, 1, 1])

    for (layer_name, loc_output), glob_output in zip(activation['Local'].items(), activation['Global'].values()):
        if layer_name in target_layer:
            loc_result.append(loc_output)
            glob_result.append(glob_output)

    del activation
    torch.cuda.empty_cache()

    return loc_result, glob_result, hooks_model_a, hooks_model_b

            

                


            

                
