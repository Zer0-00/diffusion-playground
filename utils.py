import json

def load_parameters(para_dir:str) -> dict:
    """
    loading configure json file.
    path of json file folder:./configs/
    """
    with open(para_dir, 'r') as f:
        args = json.load(f)

    if args["dataset"].lower() == "leather":
        in_channels = 3
    elif args["channels"] != "":
        in_channels = args["channels"]
    else:
        in_channels = 1

    return args