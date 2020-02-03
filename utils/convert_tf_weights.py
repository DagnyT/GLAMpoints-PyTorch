import torch
import numpy as np
from model.model import Unet_model_4
from collections import OrderedDict
import cv2

def convert_Unet_model_4_weights_tf_pytorch(weights_path=None, **kwargs):
    """
    load imported model instance
    Args:
        weights_path (str): If set, loads model weights from the given path
    """

    tf_weights = np.load(path_to_weights, allow_pickle=True).item()
    model = Unet_model_4(1)

    model_init_weights = model.state_dict()

    if weights_path:
        unrolled_tf_dictionary = unroll_tf_dictionary(tf_weights)
        converted_weights = convert_tf_weights_to_pytorch(model_init_weights, unrolled_tf_dictionary)
        model.load_state_dict(converted_weights)

        torch.save(model.state_dict(),'unet_model4_converted_tf_weights.pth')
    return model


def convert_tf_weights_to_pytorch(model_init_weights, unrolled_tf_dictionary):
    converted_weights = {}

    for key, value in model_init_weights.items():
        if 'num_batches_tracked' in key:
            converted_weights[key] = value
        elif 'conv' in key or 'decon' in key or 'final' in key:
            if len(unrolled_tf_dictionary[key].shape)>1:
                converted_weights[key] = torch.from_numpy(unrolled_tf_dictionary[key].transpose(3, 2, 0, 1).astype(np.float32))
            else:
                converted_weights[key] = torch.from_numpy(unrolled_tf_dictionary[key].astype(np.float32))

        else:
            converted_weights[key] = torch.from_numpy(unrolled_tf_dictionary[key].astype(np.float32))

    return converted_weights

def unroll_tf_dictionary(tf_weights):

    tf_unrolled_dict = OrderedDict()

    current_value = 1
    for idx, (key, value) in enumerate(tf_weights.items()):

        if 'decon' in key or 'final' in key:
             num = 0
        else:

            sub_indx = int(key.split('_')[2])
            if 'conv' in key and sub_indx == 1:
                  num  = 0
            elif 'batch' in key and sub_indx == 1:
                 num  = 1
            elif 'conv' in key and sub_indx == 2:
                num=3
            elif 'batch' in key and sub_indx == 2:
                num=4

            key = key.split('_')[0] + '_' + key.split('_')[1]

        for idx2, (subkey, subvalue) in enumerate(value.items()):

               if 'final' not in key and \
                       (int(key.split('_')[1]) > current_value):
                    current_value += 1

               new_key = key+'.'+str(num)+'.'
               if subkey == 'kernel' or subkey == 'gamma':
                   new_key = new_key+'weight'
               elif subkey == 'bias' or subkey =='beta' :
                   new_key = new_key+'bias'
               elif subkey == 'moving_mean':
                   new_key = new_key+'running_mean'
               elif subkey == 'moving_variance':
                   new_key = new_key+'running_var'


               new_key = new_key.replace('batch','conv')

               if key == 'final':
                   if subkey == 'kernel':
                       new_key = 'final.weight'
                   elif subkey == 'bias':
                       new_key = 'final.bias'

               tf_unrolled_dict[new_key] = subvalue

    return tf_unrolled_dict

if __name__ == '__main__':

    '''
    With script 'export_tf_weights_np.py' unet4.npy is created from https://gitlab.com/retinai_sandro/glampoints.  
    '''

    path_to_weights = 'weights/unet4.npy'
    model = convert_Unet_model_4_weights_tf_pytorch(path_to_weights)
    model.cuda()

    image_norm = np.ones((255,255))
    input = torch.from_numpy(image_norm).float().unsqueeze(0).unsqueeze(0).cuda()

    model.eval()
    output = model(input)
    print('')