import logging
import os

import numpy as np
import pandas as pd
import torch
from torch import nn

import time

from fedml.core import ServerAggregator

from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, f1_score, classification_report



class ExpertAggregator(ServerAggregator):

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        str_time = time.strftime("%y%m%d_%H%M%S")
        torch.save(self.model.cpu().state_dict(), os.path.join(self.args.model_file_cache_folder, f"FLmodel_{str_time}.pt"))

    def test(self, test_data, device, args):
        pass

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        model = self.model

        model.to(device)
        model.eval()
        
        tot_correct = 0
        tot_samples = 0

        criterion = nn.CrossEntropyLoss().to(device)
        
        client_train_loss = []
        client_test_loss = []
        client_train_acc = []
        client_test_acc = []

        for client_index in train_data_local_dict.keys():
            train_data = train_data_local_dict[client_index]
            
            #  print("(Train) Client", client_index)
            train_running_loss = 0
            
            for batch_idx, x in enumerate(train_data):
                
                x, stage = x[0].to(device), x[1].to(device)
                
                pred_value, _ = model(x)
                
                loss = criterion(pred_value, stage)
                train_running_loss += loss.item()
                
                pred = torch.argmax(pred_value, dim=1)
                correct = torch.sum(pred == stage)
                
                tot_correct += correct
                tot_samples += x.size(0)
                
            acc = tot_correct / tot_samples
            
            client_train_loss.append(train_running_loss / len(train_data))
            client_train_acc.append(acc)

        for client_index in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_index]
            
            test_running_loss = 0
            val_pred = []
            val_real = []
            
            print("(Test) Client",client_index)
            
            for batch_idx, x in enumerate(test_data):
                x, stage = x[0].to(device), x[1].to(device)
                
                pred_value, _ = model(x)
                
                loss = criterion(pred_value, stage)
                test_running_loss += loss.item()
                
                pred = torch.argmax(pred_value, dim=1)
                correct = torch.sum(pred == stage)
                
                tot_correct += correct
                tot_samples += x.size(0)
                val_pred.extend(pred.detach().cpu().numpy())
                val_real.extend(stage.detach().cpu().numpy())
            
            acc = tot_correct / tot_samples
            
            client_test_loss.append(test_running_loss / len(test_data))
            client_test_acc.append(acc)

        print()
        print("#################################################################")
        
        client_train_acc = np.array(client_train_acc)
        client_test_acc = np.array(client_test_acc)
        client_train_loss = np.array(client_train_loss)
        client_test_loss = np.array(client_test_loss)
                
        client_idx = 1
        
        for train_acc, test_acc, train_loss, test_loss in zip(client_train_acc, client_test_acc, client_train_loss, client_test_loss):
            print(f"Client {client_idx} train loss: {round(train_loss.mean(),3)}")
            print(f"Client {client_idx} train acc: {round(train_acc.mean(),3)}")
            print(f"Client {client_idx} test loss: {round(test_loss.mean(),3)}")
            print(f"Client {client_idx} test acc: {round(test_acc.mean(),3)}")
            print("------------------------------------------------------------")
            
            client_idx += 1       
        
        logging.info("Client train loss {}".format(client_train_loss.mean()))
        logging.info("Client train acc {}".format(client_train_acc.mean()))
        logging.info("Client test loss {}".format(client_test_loss.mean()))
        logging.info("Client test acc {}".format(client_test_acc.mean()))
        
        
        return True