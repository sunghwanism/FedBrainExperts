import sys
sys.path.append("./")

from fedml import fedml
from data.data_loader import load_data
from fedml import FedMLRunner
from model.depthwiseNet import DepthNet
from src.FedMeta.trainer.Aggregator import FedMeta_aggregator
from src.FedMeta.trainer.FedTrainer import FedMetaTrainer
import numpy as np

import torch
import random
import torch.backends.cudnn as cudnn

from fedml.arguments import Arguments, add_args
from fedml.constants import FEDML_TRAINING_PLATFORM_CROSS_SILO

_global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

def get_args():

    fedml._global_training_type = FEDML_TRAINING_PLATFORM_CROSS_SILO

    cmd_args = add_args()
    args = Arguments(cmd_args, training_type=_global_training_type,)
    # args.get_default_yaml_config(cmd_args, training_type=_global_training_type)
    args.load_yaml_config("./config/fedml_config.yaml")
    
    return args


if __name__ == "__main__":
    # init FedML framework
        
    args = get_args()
    args = fedml.init(args=args)
    
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.random_seed)

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args, length=30)

    # load model
    flmodel = DepthNet(lengths=30, patch_size=30, in_chans=2, embed_dim=256, norm_layer=None, output_dim=args.class_num)

    for param in flmodel.parameters():
        param.requires_grad = True

    # create trainer
    trainer = FedMetaTrainer(flmodel, args)
    aggregator = FedMeta_aggregator(flmodel, args)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, flmodel, trainer, aggregator)
    fedml_runner.run()
