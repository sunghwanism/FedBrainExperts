import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset

import nibabel as nib


class FLDataset(TensorDataset):

    def __init__(self, dataname, DATAPATH, config, verbose=False, _mode='train', get_info=False):
        self.IMGPATH = os.path.join(DATAPATH, "Image", dataname)
        self.mode = _mode
        if _mode == 'train':
            self.df = pd.read_csv(os.path.join(DATAPATH, "Phenotype", f"{dataname}_Phenotype_train.csv"))
        
        elif _mode == 'val':
            self.df = pd.read_csv(os.path.join(DATAPATH, "Phenotype", f"{dataname}_Phenotype_val.csv"))
        
        else:
            self.df = pd.read_csv(os.path.join(DATAPATH, "Phenotype", f"{dataname}_Phenotype_test.csv"))

        if dataname == 'DecNef':
            self.df['Subject'] = self.df['Subject'].apply(lambda x: str(x).zfill(3))

        self.get_info = get_info
        self.config = config

        if verbose:
            print(f"Dataset: {dataname} || n={len(self.df)}")

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        use_subj = self.df.iloc[idx]
        MRImage = nib.load(os.path.join(self.IMGPATH, use_subj['ImageFile'])).get_fdata()
        Age = use_subj['Age']

        # Convert the numpy array to a PyTorch tensor
        MRImage = torch.tensor(MRImage, dtype=torch.float32).unsqueeze(0)
        MRImage = self.crop_img(MRImage) # crop the image
        Age = torch.tensor(Age, dtype=torch.float32)

        if self.get_info:
            return (MRImage, Age, use_subj['Subject'], use_subj['Sex(1=m,2=f)'])

        return (MRImage, Age)
        
    def crop_img(self, img):

        # Get the original dimensions (assumed to be 3D data)
        d, h, w = img.shape[1:]  # Shape without the channel

        # Define the target crop size
        target_d, target_h, target_w = self.config.crop_size

        # Calculate the start and end indices for cropping (crop from the center)
        start_d = (d - target_d) // 2
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2

        # Perform the cropping
        img = img[:, start_d:start_d + target_d, start_h:start_h + target_h, start_w:start_w + target_w]
        
        return img
    
