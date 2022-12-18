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

from utils import load_parameters


# set environment configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# TODO: Using DDIM instead of DDPM


def ddim_ano(args):
    """
    Using DDIM to perform anomaly detection
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set in_channel


    # model setting



# TODO: Using DDPM with X_T to X_{T-K}
def partly_ddpm_ano():
    pass


if __name__ == '__main__':
    #set configuration
    torch.random.seed(1126)
    config_dir = os.path.join('.', 'configs')
    
    cfg = sys.argv[1]
    if cfg.isnumeric():
        para_dir = 'config{}'.format(cfg)
    elif cfg.endswith('.json'):
        para_dir = cfg
    else:
        para_dir = cfg + ".json"

    # parse input
    args = load_parameters(para_dir)

    ddim_ano(args)
