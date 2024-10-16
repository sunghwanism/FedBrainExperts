# FedBrainExperts

### Dependency
For installing the package for all process, you run below code:

    conda create -n AdaptFL python=3.9 -y
    conda activate AdaptFL

Install `torch=1.12.1` version with `CUDA 11.3` as below:

    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

Install other library as bellow:

    pip install -r requirements.txt
    

### Dataset
To-do





## Data Preprocessing
We train our all model with T1-weighted MRI


## Training

(1) Train Encoder for LDM

- Single GPU

    ```sh
    python script/AETrainer.py
    ```

- DDP training with Multi-GPU (example: num_gpu=2)

    ```sh
    torchrun --nproc_per_node=2 script/AETrainer_ddp.py
    ```

Caution: If you don't have wandb, add `--nowandb` as args in commend line
