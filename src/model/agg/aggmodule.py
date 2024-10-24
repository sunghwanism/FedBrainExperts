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
        elif self.agg_method == 'fedcka':
            self.FedCKA(local_weight)
        elif self.agg_method == 'fedkliep':
            self.FedKLIEP(local_weight, update_weight_per_client)
        else:
            raise NotImplementedError
        
        return self.global_model.to(self.device)
        
    def FedAvg(self, local_weight, update_weight_per_client):
        self.global_model.cpu()
        global_state_dict = self.global_model.state_dict()

        for client_idx, client_weights in local_weight.items():
            for k in global_state_dict.keys():
                if client_idx == 0:
                    global_state_dict[k] = (client_weights[k] * update_weight_per_client[client_idx]).type(global_state_dict[k].dtype)
                else:
                    global_state_dict[k] += (client_weights[k] * update_weight_per_client[client_idx]).type(global_state_dict[k].dtype)

        self.global_model.load_state_dict(global_state_dict)

    def FedProx(self, local_weight, update_weight_per_client):
        self.FedAvg(local_weight, update_weight_per_client)
        
    def MOON(self, local_weight, update_weight_per_client):
        self.FedAvg(local_weight, update_weight_per_client)

    def SCAFFOLD(self, local_weight, global_model):
        pass

    def FedCKA(self, local_weight, global_model):
        pass

    def FedRep(self, local_weight, global_model):
        pass

    def FedRepCKA(self, local_weight, global_model):
        pass

    def FedKLIEP(self, local_weight, update_weight_per_client):
        self.FedAvg(local_weight, update_weight_per_client)