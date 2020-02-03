import argparse
import logging
import tensorflow as tf
import numpy as np

# From here: https://github.com/ildoonet/tf-pose-estimation/issues/139

# Should be in GlamPoints Master code
from compute_glam_kp import Unet_model_4

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True

if __name__ == '__main__':
    """
    Use this script to just save weights of model
    """
    parser = argparse.ArgumentParser(description='Tensorflow Weights Saver')
    parser.add_argument('--weights', type=str, default='weights/model-34', help='')
    parser.add_argument('--model', type=str, default='unet4.npy', help='')

    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='image')

    inputs = tf.placeholder(tf.float32, [None, None, None, 1], name='inputs')
    training = tf.placeholder(tf.bool, name="mode")
    net = Unet_model_4(inputs, training, norm=True)
    saver = tf.train.Saver()
    weights_converted = {}
    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        # Load weights
        saver.restore(sess, args.weights)

        variables = tf.get_collection('variables')
        for var in variables:
            name = var.name
            name = name.split(':')[0]
            layer, parameter = name.split('/')
            if layer not in weights_converted.keys():
                weights_converted[layer] = dict()
            weights_converted[layer][parameter] = var.eval()

    np.save('{}.npy'.format(args.model), weights_converted)