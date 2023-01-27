import torch
import numpy as np
from matplotlib import pyplot as plt

from utils import tensor2np

def generate_heatmap_comparation(heatmap:torch.Tensor, input_image:torch.Tensor, save_dir:str,mask = None, max_score = 1):
    """generates heatmap

    Args:
        heatmap (torch.Tensor): input heatmaps of (1, 1, H, W)
        input_image (torch.Tensor): input images of (1, C, H, W)
        mask (torch.Tensor): input ground truth of (1, 1, H, W)
        save_dir(str): save directory
        max_score: maximum score
    """

    if mask == None:
        heatmap = np.squeeze(tensor2np(heatmap.squeeze(0)))
        input_image = tensor2np(input_image.squeeze(0))
        
        plt.figure(figsize=(13, 3))
        plt.subplot(2, 2, 1)
        plt.imshow(input_image)
        plt.title(f'Original image')

        plt.subplot(2, 2, 2)
        plt.imshow(heatmap, cmap='gray')
        # plt.imshow(input_image, interpolation='none')
        # plt.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='none')
        plt.colorbar(extend='both')
        plt.title('Anomaly Detection')
        
        plt.subplot(2,2,3)
        plt.hist(heatmap.flatten())
        plt.axis("on")
        plt.title("distribution histogram")
    else:
    #normalize
        heatmap = np.squeeze(tensor2np(heatmap.squeeze(0)))
        input_image = tensor2np(input_image.squeeze(0))
        mask = tensor2np(mask.squeeze(0))
        
        plt.figure(figsize=(13, 3))
        plt.subplot(2, 2, 1)
        plt.imshow(input_image)
        plt.title(f'Original image')

        plt.subplot(2, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Ground thuth')

        plt.subplot(2, 2, 3)
        plt.imshow(heatmap, cmap='gray')
        # plt.imshow(input_image, interpolation='none')
        # plt.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='none')
        plt.colorbar(extend='both')
        plt.title('Anomaly Detection')
        
        plt.subplot(2,2,4)
        plt.hist(heatmap.flatten())
        plt.axis("on")
        plt.title("distribution histogram")

        plt.clim(0, max_score)
        plt.savefig(save_dir)
    
    
    
