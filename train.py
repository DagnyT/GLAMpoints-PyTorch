import os
import argparse
import yaml
import torch
import sys
import time

from dataset import make_data_loader
from model import build_model, build_loss
from solver import build_optimizer
from logger import build_logger
from utils import settings
from utils.utils_CNN import plot_training
from tqdm import tqdm
from utils.metrics_comparison import AverageMetricsMeter
sys.path.append(".")

def do_validate(epoch, model, val_loader, loss_func, _device, nms):

    total_test_loss = 0

    model.eval()
    metrics = AverageMetricsMeter()

    for batch_idx, (img1, img1_norm, img2, img2_norm, H) in tqdm(enumerate(val_loader), total = len(val_loader)):

        img1, img1_norm, img2, img2_norm, H = img1.to(_device), img1_norm.to(_device), img2.to(_device), \
                                              img2_norm.to(_device), H.to(_device)
        kpmap1 = model(img1_norm)
        kpmap2 = model(img2_norm)

        loss, metrics_per_image, computed_reward1, mask_batch1 = loss_func(cfg, img1, img2, kpmap1, kpmap2, H, nms)
        metrics.add(metrics_per_image)
        total_test_loss += loss.item()

    model.train()
    total_test_loss = total_test_loss / len(val_loader)
    print('Total test loss: {}'.format(total_test_loss))
    print('Mean repeatability: {}'.format(metrics.repeatability/metrics.count))

    if cfg['LOGGING']['ENABLE_LOGGING']:
        tb_logger.add_scalars_to_tensorboard('Test', epoch, epoch, total_test_loss, metrics.value())

    return total_test_loss

def do_train(cfg, model, train_loader, val_loader, optimizer, loss_func, logger, tb_logger, _device):

    start_full_time = time.time()

    model.train()

    if cfg['LOGGING']['ENABLE_LOGGING']:
        logger.log_string(cfg)
    start_epoch, end_epoch = cfg['TRAINING']['START_EPOCH'], cfg['TRAINING']['END_EPOCHS']
    count = 0
    for epoch in range(start_epoch, end_epoch + 1):
        start_time = time.time()

        print('This is %d-th epoch' % epoch)
        total_train_loss = 0

        metrics = AverageMetricsMeter()
        if epoch == 0: nms = cfg['TRAINING']['NMS_EPOCH0']
        else: nms = cfg['TRAINING']['NMS_OTHERS']

        for batch_idx, (img1, img1_norm, img2, img2_norm, H) in enumerate(train_loader):

            img1, img1_norm, img2, img2_norm = img1.to(_device), img1_norm.to(_device), img2.to(_device), \
                                                  img2_norm.to(_device)
            optimizer.zero_grad()

            kpmap1 = model(img1_norm)
            kpmap2 = model(img2_norm)

            loss, metrics_per_image, computed_reward1, mask_batch1 = loss_func(cfg, img1, img2, kpmap1, kpmap2, H, nms)
            metrics.add(metrics_per_image)
            loss.backward()
            optimizer.step()
            count+=1
            if cfg['LOGGING']['ENABLE_LOGGING']:
                tb_logger.add_scalars_to_tensorboard('Train', epoch, count, loss.item(), metrics.value())

            if cfg["LOGGING"]["ENABLE_PLOTTING"] and batch_idx % cfg["TRAINING"]["PLOT_EVERY_X_BATCHES"] == 0:
                try:
                    plot_training(img1.data.cpu().numpy(), img2.data.cpu().numpy(),
                                  kpmap1.data.cpu().numpy(), kpmap2.data.cpu().numpy(),
                                  computed_reward1.data.cpu().numpy(), loss.data.cpu().numpy(), mask_batch1.data.cpu().numpy(),
                                  metrics_per_image, 0, epoch, cfg['LOGGING']['IMAGES_DIR'],
                                  name_to_save='epoch{}_batch{}.jpg'.format(epoch, batch_idx))
                except:
                    continue

            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss.item(), time.time() - start_time))
            print('Mean repeatability: {}'.format(metrics.repeatability/metrics.count))

            total_train_loss += float(loss)

        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(train_loader)))

        if epoch % cfg['LOGGING']['LOG_INTERVAL'] == 0:

            total_test_loss = do_validate(epoch, model, val_loader, loss_func, _device, nms)
            logger.log_string('test loss for epoch {} : {}\n'.format(epoch, total_test_loss))
            logger.log_string('Mean repeatability for epoch {} : {}\n'.format(epoch, metrics.repeatability/metrics.count))

            print('epoch %d total test loss = %.3f' % (epoch, total_test_loss))

        if epoch % cfg['TRAINING']['SAVE_MODEL_STEP'] == 0:
            savefilename = os.path.join(cfg['TRAINING']['MODEL_DIR'], str(epoch) + 'glampoints.tar')

            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(train_loader.dataset),
            }, savefilename)

            print('model is saved: {} - {}'.format(epoch, savefilename))

    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training GlamPoints detector')
    parser.add_argument('--path_ymlfile', type=str,default='configs/glampoints_training.yml', help='Path to yaml file.')

    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    _device = settings.initialize_cuda_and_logging(cfg)

    train_loader, val_loader = make_data_loader(cfg)

    model = build_model(cfg)
    model.to(_device)

    optimizer = build_optimizer(cfg, model)

    loss_func = build_loss(cfg)

    logger, tb_logger = build_logger(cfg)

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_func,
        logger,
        tb_logger,
        _device)