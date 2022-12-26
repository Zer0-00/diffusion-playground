import os
import sys
import utils

def calcu_ano_metric(args):
    """calculates the AUROC"""    

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