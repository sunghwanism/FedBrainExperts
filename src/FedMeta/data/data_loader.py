import logging
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
import torch.backends.cudnn as cudnn
import random

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

class applewatch:

    def __init__(self, PATH, length):
        
        self.x_data = []
        self.y_data = []
        
        data = pd.read_csv(PATH)
            
        for k in range(int(len(data)/length)):
            front_idx = int(k*length)
            post_idx = int((k+1)*length)
            
            temp = data[front_idx:post_idx]                
            # x_move = temp["x_move"].to_numpy()
            # y_move = temp["y_move"].to_numpy()
            # z_move = temp["z_move"].to_numpy()
            HR = temp["heart_rate"].to_numpy()
            activity = temp["steps"].to_numpy()
            stage = temp["psg_status"].to_numpy()[0].astype(int)
            
            if stage in [1,2,3,4]:
                stage = 1
                
            elif stage == 5:
                stage = 2
            
            # self.x_data.append(np.stack([x_move, y_move, z_move, HR, activity], axis=1))
            self.x_data.append(np.stack([HR, activity], axis=1))
            self.y_data.append(stage)
        
        self.x_data = np.array(self.x_data)
        self.y_data = np.array(self.y_data)
        
        self.x_data = torch.FloatTensor(self.x_data)
        self.y_data = torch.FloatTensor(self.y_data).long()
        
        self.x_data = self.x_data.permute(0, 2, 1)
        
        print("x_data shape: ", self.x_data.shape)
        print("y_data shape: ", self.y_data.shape, "class_num", self.y_data.unique().shape[0])
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        
        return self.x_data[idx], self.y_data[idx]


def load_data(args, length=30):

    train_data_global = list()
    test_data_global = list()
    train_data_local_dict = dict.fromkeys(range(args.client_num_in_total))
    test_data_local_dict = dict.fromkeys(range(args.client_num_in_total))
    train_data_local_num_dict = dict.fromkeys(range(args.client_num_in_total))
    train_data_num = 0
    test_data_num = 0

    if args.rank == 0:
        
        for i in range(args.client_num_in_total):
            PATH = os.path.join(args.database, f"c{i+1}_data.csv")
            
            generator = torch.Generator()
            generator.manual_seed(0)
            
            train_dataset = applewatch(PATH=PATH, length=length)
            train_data, test_data = torch.utils.data.random_split(train_dataset,
                                                                  [int(len(train_dataset)*0.8),
                                                                   len(train_dataset)-int(len(train_dataset)*0.8)],
                                                                  generator=generator)                        
            
            train_data_local_dict[i] = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True, drop_last=False
            )
            
            test_data_local_dict[i] = torch.utils.data.DataLoader(
                test_data, batch_size=args.batch_size, shuffle=False, drop_last=False
            )
            
            train_data_local_num_dict[i] = len(train_data_local_dict[i])
            train_data_num += train_data_local_num_dict[i]

    else:
        
        generator = torch.Generator()
        generator.manual_seed(0)
        
        PATH = os.path.join(args.database, f"c{args.rank}_data.csv")
        train_dataset = applewatch(PATH=PATH, length=length)
        train_data, test_data = torch.utils.data.random_split(train_dataset,
                                                                [int(len(train_dataset)*0.8),
                                                                len(train_dataset)-int(len(train_dataset)*0.8)],
                                                                generator=generator)           

        train_data_local_dict[args.rank - 1] = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, drop_last=False
        )
        
        test_data_local_dict[args.rank - 1] = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=False, drop_last=False
        )
        
        train_data_local_num_dict[args.rank - 1] = len(train_data_local_dict[args.rank - 1])
        train_data_num += train_data_local_num_dict[args.rank - 1]

    class_num = args.class_num
    
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    
    return dataset, class_num
