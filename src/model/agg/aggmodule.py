import torch
import numpy as np

from copy import deepcopy

class Aggregator:
    def __init__(self, global_model, device, config):
        self.global_model = global_model
        self.config = config
        self.device = device

    def aggregate(self, local_weight, update_weight_per_client):
        self.agg_method = self.config.agg_method.lower()

        if self.agg_method == 'fedavg':
            self.FedAvg(local_weight, update_weight_per_client)
        elif self.agg_method == 'fedprox': # same with fedavg aggregation method
            self.FedProx(local_weight, update_weight_per_client)
        elif self.agg_method == 'moon': # same with fedavg aggregation method
            self.MOON(local_weight, update_weight_per_client)
        elif self.agg_method == 'scaffold':
            self.SCAFFOLD(local_weight)
        elif self.agg_method == 'FedCKA':
            self.FedCKA(local_weight)
        else:
            raise NotImplementedError
        
        return self.global_model.to(self.device)

    def FedAvg(self, local_weight, update_weight_per_client):
        self.global_model.cpu()
        global_state_dict = self.global_model.state_dict()

        for k in global_state_dict.keys():
            global_state_dict[k] = local_weight[0][k] # / update_weight_per_client[0]

        for client_idx, client_weights in local_weight.items():
            if client_idx == 0:
                continue

            for k in global_state_dict.keys():
                global_state_dict[k] += client_weights[k] # / update_weight_per_client[client_idx]

        global_state_dict = {k: v / len(local_weight.keys()) for k, v in global_state_dict.items()}
        self.global_model.load_state_dict(global_state_dict)

    def FedProx(self, local_weight, update_weight_per_client):
        return self.FedAvg(local_weight, update_weight_per_client)
        
    def MOON(self, local_weight, update_weight_per_client):
        return self.FedAvg(local_weight, update_weight_per_client)

    def SCAFFOLD(self, local_weight, global_model):
        pass

    def FedCKA(self, local_weight, global_model):
        pass