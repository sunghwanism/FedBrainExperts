import torch



class Aggregator:
    def __init__(self, global_model, config):
        self.global_model = global_model
        self.config = config

    def aggregate(self, local_weight, num_client_dict, update_weight_per_client):
        self.config.agg_method = self.config.agg_method.lower()

        if self.config.agg_method == 'fedavg':
            self.FedAvg(local_weight, num_client_dict, update_weight_per_client)
        elif self.config.agg_method == 'fedprox':
            self.FedProx(local_weight)
        elif self.config.agg_method == 'moon':
            self.MOON(local_weight)
        elif self.config.agg_method == 'scaffold':
            self.SCAFFOLD(local_weight)
        elif self.config.agg_method == 'FedCKA':
            self.FedCKA(local_weight)
        else:
            raise NotImplementedError
        
        return self.global_model

    def FedAvg(self, local_weight, num_client_dict, update_weight_per_client):

        for k in self.global_model.state_dict().keys():
            if 'num_batches_tracked' in k:
                self.global_model.state_dict()[k] = int(sum(num_client_dict.values()) // self.config.batch_size)
                continue

            for client_idx, loc_w in local_weight.items():
                if client_idx == 0:
                    self.global_model.state_dict()[k] = loc_w[k] / update_weight_per_client[client_idx]
                else:
                    self.global_model.state_dict()[k] += loc_w[k] / update_weight_per_client[client_idx]

    def FedProx(self, local_weight, global_model):
        pass

    def MOON(self, local_weight, global_model):
        pass

    def SCAFFOLD(self, local_weight, global_model):
        pass

    def FedCKA(self, local_weight, global_model):
        pass