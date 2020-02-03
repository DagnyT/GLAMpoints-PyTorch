import numpy as np
import torch
import argparse
import yaml
import torch
import os
from model import build_model
import cv2
import sys
from tqdm import tqdm
from model.glampoints import non_max_suppression
from model.glampoints import sift

sys.path.append(".")

def evaluate(opt, cfg, model, name):

    images = open(opt.path_images).read().splitlines()
    images = [os.path.join(opt.path_hpatches, x) for x in images]

    SIFT = cv2.xfeatures2d.SIFT_create()

    for img_path in tqdm(images):

        img = cv2.imread(img_path, 0)

        H, W = img.shape

        img_norm = img/img.max()

        img_norm_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).cuda()

        kp_map = model(img_norm_tensor)
        kp_map_nonmax = non_max_suppression(kp_map.data.cpu().numpy().squeeze(), cfg['TEST']['NMS'], cfg['TEST']['MIN_PROB'])

        keypoints_map = np.where(kp_map_nonmax > 0)
        kp_array = np.array([keypoints_map[1], keypoints_map[0]]).T
        kp_cv2 = [cv2.KeyPoint(kp_array[i, 0], kp_array[i, 1], 10) for i in range(len(kp_array))]
        kp, desc = sift(SIFT, np.uint8(img), kp_cv2)
        kp = np.array([m.pt for m in kp], dtype=np.int32)

        outpath = img_path + '.' + name
        print(f"Saving {len(kp)} keypoints to {outpath}")

        np.savez(open(outpath,'wb'),
            imsize = (W,H),
            keypoints = kp,
            descriptors = desc,
            scores = np.zeros(len(kp)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training GlamPoints detector')
    parser.add_argument('--path_ymlfile', type=str,default='configs/glampoints_eval.yml', help='Path to yaml file.')
    parser.add_argument('--path_images', type=str,default='eval/image_list_hpatches_sequences.txt', help='Path to hpatches images.')
    parser.add_argument('--path_hpatches', type=str,default='', help='Path to hpatches images.')
    parser.add_argument('--init_weights', type=str, default='modified', help='Weights converted from GlamPoints gitlab repo')
    parser.add_argument('--name', type=str, default='glampoints_retina', help='Weights converted from GlamPoints gitlab repo')

    print('Evaluate GlamPoints on HPatches dataset')

    opt = parser.parse_args()

    with open(opt.path_ymlfile, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    name = opt.name
    model = build_model(cfg)
    if opt.init_weights == 'init':
        model.load_state_dict(torch.load(cfg['TEST']['WEIGHTS']))
    else:
        model.load_state_dict(torch.load(cfg['TEST']['WEIGHTS'])['state_dict'])

    model.cuda()
    model.eval()

    evaluate(opt, cfg, model, name)