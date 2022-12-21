import json
import os
from collections import defaultdict


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
                args[arg_name] = args[arg_name]

    #set input channel
    if args["dataset"].lower() == "leather":
        args["in_channels"] = 3
    elif args["in_channels"] == '':
        args["in_channels"] = 1

    #set input path
    if args["input_path"] == "":
        if args["dataset"] == "leather":
            args["input_path"] = os.path.join("~", "Datasets", "leather")


    #set output path
    set_default_value("output_path", os.path.join('.','output', cfgs_name))
 
    #set Unet structure
    default_Unet = {

    }
    set_default_value("layers_per_block", 2)
    set_default_value("block_out_channels", (128, 128, 256, 256, 512, 512))
    set_default_value()

    return args



def create_folders(f_dir):
    if not os.path.exists(f_dir):
        os.makedirs(f_dir)
