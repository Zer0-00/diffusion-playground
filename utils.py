import json
import os
from collections import defaultdict
from diffusers import EMAModel, UNet2DModel
import torch
from torchvision import utils as vutils
import copy
import numpy as np
import csv

def load_parameters(para_dir:str) -> dict:
    """
    loading configure json file.
    path of json file folder:./configs/
    """

    cfgs_name = os.path.basename(para_dir)[:-5]
    print("configurations:"+cfgs_name)
    with open(para_dir, 'r') as f:
        args_dict = json.load(f)

    args = defaultdict(str)
    args.update(args_dict)

    args["cfgs_name"] = cfgs_name

    def set_default_value(pairs):
    #pairs: {(arg_name, default_value)}
        for arg_name in pairs:
            if args[arg_name] == '':
                args[arg_name] = pairs[arg_name]

    #set input channel
    if args["dataset"].lower() == "leather":
        args["in_channels"] = 3
    elif args["in_channels"] == '':
        args["in_channels"] = 1

    #set input path
    if args["input_path"] == "":
        if args["dataset"] == "leather":
            args["input_path"] = os.path.join("home", "xuehong","Datasets", "MVTEC","leather")


    #set output path
    set_default_value({"output_path": os.path.join('.','output', cfgs_name)})
 
    #set default Unet structure
    default_Unet ={
        "layers_per_block": 2,
        "block_out_channels": (128, 128, 256, 256, 512, 512),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        )   
    }
    
    set_default_value(default_Unet)
    
    #set default noise scheduler
    default_noise_scheduler = {
        "num_train_timesteps": 1000,
        "beta_schedule": "linear",
        "prediction_type":"epsilon"
    }
    
    set_default_value(default_noise_scheduler)
    
    #set default training process()
    
    default_training_process ={
        "learning_rate": 1e-4,
        "max_epoch": 3000,
        "start_epoch":0,
        "checkpoint": None,
        "save_epoch": 1000,
        "exhibit_epoch":200, 
        "seed": 1126,
        "batch_size": 4,
        #optimizer
        "weight_decay": 0.0,
        "betas": [0.9, 0.999],
        "optimiser_epsilon": 1e-8,
        #Dataloader
        "shuffle": True,
        "drop_last": True,
        #EMA model
        "ema_inv_gamma": 1.0,
        "ema_power": 2/3,
        #set drop_rate to 0.9999
        "ema_min_value": 0.9999,
        "ema_max_value": 0.9999
    }
    set_default_value(default_training_process)
    return args



def create_folders(f_dir):
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)

def save_checkpoint(
    output_dir: str,
    model: UNet2DModel,
    ema_model: EMAModel,
    scheduler,
    optimizer:torch.optim.Optimizer,
    start_epoch: int
):
    """
        Save checkpoint to current device
    """
    #transfer to cpu
    save_model = copy.deepcopy(model)
    save_model.to("cpu")
    save_ema_model = copy.deepcopy(ema_model)
    save_ema_model.averaged_model.to("cpu")
    
    #save
    checkpoint = {}
    checkpoint["unet"] = save_model.state_dict()
    checkpoint["ema_model"] = save_ema_model.averaged_model.state_dict()
    checkpoint["ema_optimization_step"] = save_ema_model.optimization_step
    checkpoint["scheduler_config"] = scheduler.config
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["start_epoch"] = start_epoch
    
    torch.save(checkpoint, output_dir)

def save_metrics(metrics, file_dir):
    #save metrics
    with open(file_dir, 'w') as f:
        metrics_name = ""
        for name in metrics:
            metrics_name += name
            metrics_name += ","
        metrics_name = metrics_name[:-1] + '\n'
        f.write(metrics_name)
        for metric in metrics:
            f.write("{:.4f} +- {:.4f}".format(np.mean(metrics[metric]), np.std(metrics[metric])))

def save_images(images, images_folder_path):
    for images_name in images:
        imgs = images[images_name]
        
        imgs_folder = os.path.join(images_folder_path, images_name)
        create_folders(imgs_folder)
        
        for idx, img in enumerate(imgs):
            f_dir = os.path.join(imgs_folder, "{}.jpg".format(idx))
            vutils.save_image(img, f_dir)
            
def save_detail_metrics(metrics:dict, file_path):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = list(metrics.keys())
        writer.writerow(headers)
        #write rows
        for idx,_ in enumerate(metrics[headers[0]]):
            row = []
            for header in headers:
                row.append(metrics[header][idx])
            writer.writerow(row)

def tensor2np(input_image:torch.Tensor, normalize=False):
    """
    change input tensor(C,H,W) to cv2 np.array(list(np.array(np.uint8)))
    """
    if normalize:
        input_image = (input_image - input_image.min() / input_image.max()-input_image.min())
        input_image = torch.permute(input_image, (1,2,0)).detach().cpu().numpy()
        input_image = (input_image * 255).astype(np.uint8)
        
    else:
        input_image = torch.permute(input_image, (1,2,0)).detach().cpu().numpy()
    return input_image

def normalize_image(input_images:torch.Tensor):
    """
    normalize batch image to (0,1) for every image in batch
    """
    pixel_dim = list(range(2, len(input_images.shape)))
    picture_shape = input_images.shape[2:]
    
    maxs, _ = torch.max(input_images.reshape(input_images.shape[0], input_images.shape[1], -1), dim=-1)
    maxs = maxs.reshape(*maxs.shape, *((1,)*len(picture_shape)))
    mins, _ = torch.min(input_images.reshape(input_images.shape[0], input_images.shape[1], -1), dim=-1)
    mins = mins.reshape(*mins.shape, *((1,)*len(picture_shape)))

    normalized_images = (input_images - mins.repeat(1,1,*picture_shape)) / (maxs-mins).repeat(1,1,*picture_shape)
    
    return normalized_images

