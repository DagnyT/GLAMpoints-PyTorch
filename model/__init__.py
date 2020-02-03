from .model_modified import Unet_model_4 as Unet_model_4_modified
from .model import Unet_model_4
from .loss import Reward_Loss

def build_model(cfg):

    if cfg['TRAINING']['MODEL'] == 'init':
        model = Unet_model_4(cfg['TRAINING']['INPUT_LAYER'])
    else:
        model = Unet_model_4_modified(cfg['TRAINING']['INPUT_LAYER'])

    return model

def build_loss(cfg):

    loss = Reward_Loss(cfg)
    return loss