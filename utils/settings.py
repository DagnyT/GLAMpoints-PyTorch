import torch
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn

def initialize_cuda_and_logging (cfg):

    print(("NOT " if not cfg['TRAINING']['CUDA'] else "") + "Using cuda")

    # add experiment name to the folder names logs/experiments_name/logs/, logs/experiments_name/model/, etc.
    LOG_DIR = os.path.join(cfg['LOGGING']['LOG_DIR'], cfg['TRAINING']['EXPERIMENT_NAME'])
    MODELS_DIR = os.path.join(cfg['LOGGING']['LOG_DIR'],cfg['TRAINING']['EXPERIMENT_NAME'],cfg['TRAINING']['MODEL_DIR'])
    IMAGES_DIR = os.path.join(cfg['LOGGING']['LOG_DIR'], cfg['TRAINING']['EXPERIMENT_NAME'],cfg['LOGGING']['IMG_DIR'],)

    cfg['LOGGING']['LOG_DIR'] = LOG_DIR
    cfg['TRAINING']['MODEL_DIR'] = MODELS_DIR
    cfg['LOGGING']['IMAGES_DIR'] = IMAGES_DIR

    if cfg['TRAINING']['CUDA']:
        if not torch.cuda.is_available():
            raise Exception("CUDA is NOT available!")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['TRAINING']['GPU_ID'])

        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg['TRAINING']['SEED'])
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(cfg['TRAINING']['SEED'])
      #  torch.backends.cudnn.benchmark = False

    # create logging directory

    if not os.path.isdir(LOG_DIR):  os.makedirs(LOG_DIR)
    if not os.path.isdir(MODELS_DIR):  os.makedirs(MODELS_DIR)
    if not os.path.isdir(IMAGES_DIR):  os.makedirs(IMAGES_DIR)

    # set random seeds
    random.seed(cfg['TRAINING']['SEED'])
    np.random.seed(cfg['TRAINING']['SEED'])

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
