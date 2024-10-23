
import argparse


def FLconfig():
    parser = argparse.ArgumentParser()

    ####################### BASE Configuration #######################
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # PATH
    # '/data/msh2044/AdaptFL/'
    # '/NFS/Users/moonsh/AdaptFL/'
    parser.add_argument('--base_path', type=str, default='/NFS/Users/moonsh/AdaptFL/',)
    
    # Data
    # '/local_datasets/msh2044/FLData/'
    # '/NFS/Users/moonsh/data/FLData/'
    parser.add_argument('--data_path', type=str, default='/NFS/Users/moonsh/data/FLData/',
                        help='Path to data')
    
    parser.add_argument('--crop_size',type=int, nargs='+', default=(96, 128, 96),)
    
    # Device Arguments
    parser.add_argument('--device_id', type=int, help='Which GPU to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    

    ####################### Train Configuration #######################
    # Train Arguments
    parser.add_argument('--batch_size', type=int, default=45,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=5,    
                        help='Number of epochs per round')
    parser.add_argument('--lr', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer (sgd, adam)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    
    # Federated Learning Arguments
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--round', type=int, default=100,
                        help='Number of rounds')
    parser.add_argument('--data_idx', type=int, nargs='+', 
                        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='Index of the dataset to use')
    parser.add_argument('--agg_method', type=str,
                        help='Aggregation method (FedAvg, FedProx, MOON, SCAFFOLD, FedCKA)')
    parser.add_argument('--proximal_mu', type=float, default=1,
                        help='Proximal term for FedProx')
    parser.add_argument('--contrastive_temp', type=float, default=0.5,
                        help='Contrastive temperature for MOON')
    parser.add_argument('--warmup', type=float, default=0,
                        help='warmup for MOON')

    ####################### MODEL Configuration ####################### 

    parser.add_argument('--model', type=str, default='resnet',
                        help='Which model to use')
    parser.add_argument('--model_depth', type=int, default=34,
                        help='Depth of model')
    parser.add_argument('--out_dim', type=int, default=1, # 1 for regression
                        help='Output dimension')
    parser.add_argument('--personalized', action='store_true',
                        help='Personalized model')

    ####################### SAVE Configuration #######################

    # Save and Log Arguments
    # '/data/msh2044/AdaptFL/ckpt/'
    # '/NFS/Users/moonsh/AdaptFL/ckpt/'
    parser.add_argument('--save_path', type=str, default='/NFS/Users/moonsh/AdaptFL/ckpt/',
                        help='Where to save the model')
    parser.add_argument('--local_log_interval', type=int, default=5,)
    
    ####################### Wandb Configuration #######################
    # Wandb Arguments
    parser.add_argument('--wandb_project', type=str, help='Wandb project')
    parser.add_argument('--wandb_entity', type=str, default='msh2044',
                        help='Wandb entity')
    parser.add_argument('--nowandb', action='store_true',
                        help='Don\'t use wandb')
    
    return parser
