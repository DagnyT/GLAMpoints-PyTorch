import torch

def build_optimizer(cfg, model):

    if cfg['TRAINING']['OPTIMIZER'] == 'sgd':
        optimizer = getattr(torch.optim, cfg['TRAINING']['OPTIMIZER'])(model.parameters(), momentum=cfg['TRAINING']['MOMENTUM'])
    else:
        optimizer = getattr(torch.optim, cfg['TRAINING']['OPTIMIZER'])(model.parameters(), lr=cfg['TRAINING']['BASE_LR'], betas=(0.9, 0.999))

    return optimizer
