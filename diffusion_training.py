"""
varying testing models
"""

from matplotlib import pyplot as plt
import numpy as np
import os
# set environment configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import sys
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.training_utils import EMAModel

import utils
import dataset
from models import AnomalyDetectionModel



def training(args):
    #basic configuration
    torch.manual_seed(args["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for folder in ["metrics", "images", "checkpoint"]:
        f_dir = os.path.join(args["output_path"], folder)
        utils.create_folders(f_dir)
    
    del folder, f_dir

    writer = SummaryWriter(log_dir=os.path.join(args["output_path"],"metrics"))
    
    #initialize dataset
    if args["dataset"] == "leather":
        rgb = (args["in_channels"] == 3)
        train_dataset = dataset.MVtec_Leather(
            args["input_path"],
            anomalous=False,
            img_size=args["img_size"],
            rgb=rgb
            )
        # test_dataset = MVtec_Leather(
        #     args["input_path"],
        #     anomalous=True,
        #     img_size=args["img_size"],
        #     rgb=rgb,
        #     include_good=True
        # )
        del rgb
        
    elif args["dataset"].lower() == "chexpert":
        train_dataset = dataset.CheXpert(
            args["input_path"],
            anomalous=False
        )
        
    elif args["dataset"].lower() == "brats2020":
        train_dataset = dataset.Brats2020(
            args["input_path"],
            anomalous=False
        )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args["batch_size"],shuffle=args["shuffle"], drop_last=args["drop_last"]
        )
    # test_dataloader = DataLoader(
    #     test_dataset,batch_size=args["batch_size"],shuffle=args["shuffle"], drop_last=args["drop_last"]
    #     )


    #initialize the model
    start_epoch = 0
    
    model = UNet2DModel(
        sample_size=args["img_size"],
        in_channels=args["in_channels"],
        out_channels=args["in_channels"],
        layers_per_block=args["layers_per_block"],
        block_out_channels=args["block_out_channels"],
        down_block_types=args["down_block_types"],
        up_block_types=args["up_block_types"]
    )
    model.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args["num_train_timesteps"],
        beta_schedule=args["beta_schedule"],
        prediction_type=args["prediction_type"]
    )
        
    #initalize EMAmodel
    ema_model = EMAModel(
        model,
        inv_gamma=args["ema_inv_gamma"],
        power=args["ema_power"],
        max_value=args["ema_max_value"],
        min_value=args["ema_min_value"]
    )
    
    #initialize optimiser
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["learning_rate"],
        betas=args["betas"],
        weight_decay=args["weight_decay"],
        eps=args["optimiser_epsilon"]
    )
    #load_checkpoint
    if args["checkpoint"] is not None:
        #find supposed checkpoint
        if args["checkpoint"] != "latest":
            checkpoint_path = os.path.join(args["output_path"], "checkpoint",args["checkpoint"]+".pt")
        else:
            folder = os.path.join(args["output_path"], "checkpoint")
            candidates = [cp for cp in os.listdir(folder) if cp.endswith(".pt")]
            last = sorted(candidates)[-1]
            checkpoint_path = os.path.join(folder, last)
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        #resume model
        model.load_state_dict(checkpoint["unet"])
        ema_model.averaged_model.load_state_dict(checkpoint["ema_model"])
        ema_model.optimization_step = checkpoint["ema_optimization_step"]
        noise_scheduler.from_config(checkpoint["scheduler_config"])
        #resume optimiser
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        start_epoch = checkpoint["start_epoch"]
        
        del checkpoint
        
    #training 
    for epoch in range(start_epoch, args["max_epoch"]):
        print("epoch: {}/{}".format(epoch, args["max_epoch"]))
        model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader): 
            input_images = batch["input"]
            input_images = input_images.to(device)
            
            #sample noise and timesteps
            noise = torch.randn(input_images.shape).to(input_images.device)
            
            batch_size = input_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=input_images.device
                ).long()
            
            #add noise
            noisy_images = noise_scheduler.add_noise(input_images, noise, timesteps)
            
            #denoising
            epsilons = model(noisy_images, timesteps).sample
            
            #update weights
            loss = F.mse_loss(epsilons, noise,reduction="mean")
            
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            ema_model.step(model)
            
            mean_loss += loss.detach()
            
        mean_loss /= (step + 1)
        
        #report and save results
        writer.add_scalar("train_loss", mean_loss, epoch)
        
        #sample 1 batch of image and show the noising and denoising result
        if epoch % args["exhibit_epoch"] == 0 or epoch == (args["max_epoch"] - 1):
            writer.add_images("input images", utils.normalize_image(input_images), epoch)
            
            ano_detect = AnomalyDetectionModel(ema_model.averaged_model, noise_scheduler)
            generator = torch.Generator(device=ano_detect.device).manual_seed(args["seed"])
            
            recovered = ano_detect.generate_from_scratch(
                input_images=input_images,
                generator=generator,
                time_steps=noise_scheduler.config.num_train_timesteps - 1
            )[0]
            
            writer.add_images("recovered images", utils.normalize_image(recovered), epoch)
            
            del ano_detect,generator,recovered
        
        #save checkpoint    
        if epoch % args["save_epoch"] == 0 or epoch == (args["max_epoch"] - 1):
            checkpoint_dir = os.path.join(args["output_path"], "checkpoint", "{}.pt".format(epoch))
            #now save checkpoint with cpu (be cautious when load -> use map_location)
            utils.save_checkpoint(
                checkpoint_dir,
                model,
                ema_model,
                noise_scheduler,
                optimizer,
                epoch
            )
            
    writer.close()
            
        

    

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

    config_dir = os.path.join('.', 'configs')
    
    cfg = sys.argv[1]
    if cfg.isnumeric():
        para_name = 'configs{}.json'.format(cfg)
    elif cfg.endswith('.json'):
        para_name = cfg
    else:
        para_name = cfg + ".json"
    para_dir = os.path.join(config_dir, para_name)
    
    # parse input
    args = utils.load_parameters(para_dir)

    training(args)

