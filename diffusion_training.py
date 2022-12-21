"""
varying testing models
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import random
import sys
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from diffusers import UNet2DModel

import utils


# set environment configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def training(args):
    #basic configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for folder in ["metric", "images", "model"]:
        f_dir = os.path.join(args["output_dir"], folder)
        utils.create_folders(f_dir)
    
    del folder, f_dir

    #initial models
    writer = SummaryWriter()

    model = UNet2DModel(
        sample_size=args["resolution"],
        in_channels=args["in_channels"],
        out_channels=args["in_channels"]

    )

    

    

# TODO: Using DDIM instead of DDPM
def ddim_ano(args):
    """
    Using DDIM to perform anomaly detection
    """
    pass

# TODO: Using DDPM with X_T to X_{T-K}
def partly_ddpm_ano():
    pass


if __name__ == '__main__':
    #set configuration
    torch.random.seed(1126)
    config_dir = os.path.join('.', 'configs')
    
    cfg = sys.argv[1]
    if cfg.isnumeric():
        para_name = 'configs{}'.format(cfg)
    elif cfg.endswith('.json'):
        para_name = cfg
    else:
        para_name = cfg + ".json"
    para_dir = os.path.join(config_dir, para_name)
    
    # parse input
    args = utils.load_parameters(para_dir)

    training(args)

