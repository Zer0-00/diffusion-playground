import json
import os
from collections import defaultdict
from diffusers import EMAModel, UNet2DModel
import torch

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
    model.to("cpu")
    ema_model.averaged_model.to("cpu")
    
    #save
    checkpoint = {}
    checkpoint["unet"] = model.state_dict()
    checkpoint["ema_model"] = ema_model.averaged_model.state_dict()
    checkpoint["ema_optimization_step"] = ema_model.optimization_step
    checkpoint["scheduler_config"] = scheduler.config
    checkpoint["optimizer"] = optimizer.state_dict()
    checkpoint["start_epoch"] = start_epoch
    
    torch.save(checkpoint, output_dir)
