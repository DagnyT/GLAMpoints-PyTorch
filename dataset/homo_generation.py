#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:35:26 2019

@author: truongp
"""
import numpy as np
import cv2
import random


def homography_sampling(shape, parameters, seed=None):
    """Sample a random valid homography, as a composition of translation, rotation,
    scaling, shearing and perspective transforms.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        parameters: dictionnary containing all infor on the transformations to apply.
        ex:
        parameters={}
        scaling={'use_scaling':True, 'min_scaling_x':0.7, 'max_scaling_x':2.0, \
                 'min_scaling_y':0.7, 'max_scaling_y':2.0}
        perspective={'use_perspective':False, 'min_perspective_x':0.000001, 'max_perspective_x':0.0009, \
                  'min_perspective_y':0.000001, 'max_perspective_y':0.0009}
        translation={'use_translation':True, 'max_horizontal_dis':100, 'max_vertical_dis':100}
        shearing={'use_shearing':True, 'min_shearing_x':-0.3, 'max_shearing_x':0.3, \
                  'min_shearing_y':-0.3, 'max_shearing_y':0.3}
        rotation={'use_rotation':True, 'max_angle':90}
        parameters['scaling']=scaling
        parameters['perspective']=perspective
        parameters['translation']=translation
        parameters['shearing']=shearing
        parameters['rotation']=rotation
    Returns:
        A 3x3 matrix corresponding to the homography transform.
    """
    # if seed is not None:
    #     random.seed(seed)
    if parameters['ROTATION']['USE_ROTATION']:
        (h, w) = shape
        center = (w // 2, h // 2)
        y = random.randint(-parameters['ROTATION']['MAX_ANGLE'], \
                           parameters['ROTATION']['MAX_ANGLE'])
        # perform the rotation
        M = cv2.getRotationMatrix2D(center, y, 1.0)
        homography_rotation = np.concatenate([M, np.array([[0, 0, 1]])], axis=0)
    else:
        homography_rotation = np.eye(3)

    if parameters['TRANSLATION']['USE_TRANSLATION']:
        tx = random.randint(-parameters['TRANSLATION']['MAX_HORIZONTAL_DIS'], \
                            parameters['TRANSLATION']['MAX_HORIZONTAL_DIS'])
        ty = random.randint(-parameters['TRANSLATION']['MAX_VERTICAL_DIS'], \
                            parameters['TRANSLATION']['MAX_VERTICAL_DIS'])
        homography_translation = np.eye(3)
        homography_translation[0, 2] = tx
        homography_translation[1, 2] = ty
    else:
        homography_translation = np.eye(3)

    if parameters['SCALING']['USE_SCALING']:
        scaling_x = random.choice(np.arange(parameters['SCALING']['MIN_SCALING_X'], \
                                            parameters['SCALING']['MAX_SCALING_X'], 0.1))
        scaling_y = random.choice(np.arange(parameters['SCALING']['MIN_SCALING_Y'], \
                                            parameters['SCALING']['MAX_SCALING_Y'], 0.1))
        homography_scaling = np.eye(3)
        homography_scaling[0, 0] = scaling_x
        homography_scaling[1, 1] = scaling_y
    else:
        homography_scaling = np.eye(3)

    if parameters['SHEARING']['USE_SHEARING']:
        shearing_x = random.choice(np.arange(parameters['SHEARING']['MIN_SHEARING_X'], \
                                             parameters['SHEARING']['MAX_SHEARING_X'], 0.0001))
        shearing_y = random.choice(np.arange(parameters['SHEARING']['MIN_SHEARING_Y'], \
                                             parameters['SHEARING']['MAX_SHEARING_Y'], 0.0001))
        homography_shearing = np.eye(3)
        homography_shearing[0, 1] = shearing_y
        homography_shearing[1, 0] = shearing_x
    else:
        homography_shearing = np.eye(3)

    if parameters['PERSPECTIVE']['USE_PERSPECTIVE']:
        perspective_x = random.choice(np.arange(parameters['PERSPECTIVE']['MIN_PERSPECTIVE_X'], \
                                                parameters['PERSPECTIVE']['MAX_PERSPECTIVE_X'], 0.00001))
        perspective_y = random.choice(np.arange(parameters['PERSPECTIVE']['MIN_PERSPECTIVE_Y'], \
                                                parameters['PERSPECTIVE']['MAX_PERSPECTIVE_Y'], 0.00001))
        homography_perspective = np.eye(3)
        homography_perspective[2, 0] = perspective_x
        homography_perspective[2, 1] = perspective_y
    else:
        homography_perspective = np.eye(3)

    homography = np.matmul(np.matmul(np.matmul(np.matmul(homography_rotation, homography_translation), \
                                               homography_shearing), homography_scaling), \
                           homography_perspective)
    return homography
