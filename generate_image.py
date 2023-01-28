import torch
import numpy as np
from matplotlib import pyplot as plt

from utils import tensor2np

def generate_heatmap_comparation(heatmap:torch.Tensor, input_image:torch.Tensor, save_dir:str,generated_image = None, mask = None, max_score = 2):
    """generates heatmap

    Args:
        heatmap (torch.Tensor): input heatmaps of (1, 1, H, W)
        input_image (torch.Tensor): input images of (1, C, H, W)
        mask (torch.Tensor): input ground truth of (1, 1, H, W)
        save_dir(str): save directory
        max_score: maximum score
    """
    tt_images = 4 + (mask is not None) + (generated_image is not None)
    
    rows = int(tt_images / 2) + 1
    cnt = 1
    
    input_image = tensor2np(input_image.squeeze(0))
    
    plt.figure(figsize=(13, rows * 2.2 + 1))
    plt.subplot(rows, 2, cnt)
    plt.imshow(input_image, cmap="gray")
    plt.title(f'Original image')
    cnt += 1
    
    if mask is not None:
        mask = tensor2np(mask.squeeze(0))
        plt.subplot(rows, 2, cnt)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Ground thuth')
        cnt += 1
    
    if generated_image is not None:
        generated_image = np.squeeze(tensor2np(generated_image.squeeze(0)))
        plt.subplot(rows, 2, cnt)
        plt.imshow(generated_image, cmap='gray')
        plt.title('generated')
        cnt += 1
        
    heatmap = np.squeeze(tensor2np(heatmap.squeeze(0)))
    plt.subplot(rows, 2, cnt)
    plt.imshow(heatmap, cmap='gray')
    plt.colorbar(extend='both')
    plt.title('MSE')
    cnt += 1
    
    anomaly_map = heatmap > 0.5
    plt.subplot(rows, 2, cnt)
    plt.imshow(anomaly_map, cmap='gray')
    plt.colorbar(extend='both')
    plt.title('anamaly map')
    cnt += 1
    
    plt.subplot(rows, 2, cnt)
    plt.hist(heatmap.flatten())
    plt.axis("on")
    plt.title("distribution histogram")
    
    plt.clim(0, max_score)
    plt.savefig(save_dir)
    plt.close()
    
    
    
