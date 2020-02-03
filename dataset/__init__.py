import torch
from .synthetic_dataset import SyntheticDataset

def make_data_loader(cfg):

    num_workers = cfg['TRAINING']['NUM_WORKERS']

    if (cfg['TRAINING']['TRAINING_SET'] == 'ps-dataset'):

        train_set = SyntheticDataset(cfg, train=True)
        val_set   = SyntheticDataset(cfg, train=False)

    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_set, drop_last=True,
                                               batch_size=cfg['TRAINING']['BATCH_SIZE'],
                                               shuffle=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(val_set, drop_last=True,
                                             batch_size=cfg['TEST']['BATCH_SIZE'],
                                             shuffle=False,
                                             num_workers=num_workers)

    return train_loader, val_loader