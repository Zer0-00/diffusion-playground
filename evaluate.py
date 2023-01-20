import os
import sys
import torch
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from diffusers import UNet2DModel,EMAModel, DDPMScheduler

import utils
from dataset import MVtec_Leather
from models import AnomalyDetectionModel
from generate_image import generate_heatmap_comparation

def calcu_ano_metric(args):
    """calculates the AUROC"""    
    args["batch_size"] = 1
    #basic configuration
    torch.manual_seed(args["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for folder in ["metric", "images"]:
        f_dir = os.path.join(args["output_path"], folder)
        utils.create_folders(f_dir)
    
    del folder, f_dir
    
    #initialize dataset
    rgb = (args["in_channels"] == 3)

    test_dataset = MVtec_Leather(
        args["input_path"],
        anomalous=True,
        img_size=args["img_size"],
        rgb=rgb,
        include_good=True,
        prepare=("",)
    )

    test_dataloader = DataLoader(
        test_dataset,batch_size=args["batch_size"],shuffle=args["shuffle"], drop_last=args["drop_last"]
        )

    del rgb
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
    
    #load_checkpoint and create pipeline
    
    #find supposed checkpoint
    if args["checkpoint"] != "latest":
        checkpoint_path = os.path.join(args["output_path"], "model",args["checkpoint"]+".pt")
    else:
        folder = os.path.join(args["output_path"], "model")
        candidates = [cp for cp in os.listdir(folder) if cp.endwith(".pt")]
        last = sorted(candidates)[-1]
        checkpoint_path = os.path.join(folder, last)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
        
    #resume model
    ema_model.averaged_model.load_state_dict(checkpoint["ema_model"])
    noise_scheduler.from_config(checkpoint["scheduler_config"])

    del checkpoint
    
    ano_detect = AnomalyDetectionModel(unet=ema_model.averaged_model, scheduler=noise_scheduler)
    ano_detect.unet.eval()
    total_step = len(test_dataset)
    
    auroc = []
    images_folder_path = os.path.join(args["output_path"], "images")
    for step, batch in test_dataloader:
        print("step:{}/{}".format(step, total_step))
        
        input_images = batch["input"]
        input_images = input_images.to(device)
        mask = batch["mask"]
        generator = torch.Generator(device=ano_detect.device).manual_seed(args["seed"])
        
        heatmap = ano_detect.generate_mse_detection_map(
            input_images=input_images,
            generator=generator,
            time_steps= noise_scheduler.config.num_train_timesteps - 1
        )

        auroc.append(calcu_AUROC(mask, heatmap))
        img_save_dir = os.path.join(images_folder_path, "{}.jpg".format(step))
        generate_heatmap_comparation(heatmap, input_images, mask, img_save_dir)
        
    metrics = {
        "AUROC": auroc
    }

    
    #save the metrics and images
    file_path = os.path.join(args["output_path"], "metrics", "mse_method.csv")
    utils.save_metrics(metrics, file_path)
    

def calcu_AUROC(mask:torch.Tensor, heatmap:torch.Tensor):
    return roc_auc_score(mask.detach().cpu().numpy().flatten(), heatmap.detach().cpu().numpy().flatten())
    


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

    calcu_ano_metric(args)