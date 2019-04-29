from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from networks import *
import argparse
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()

parser.add_argument('--log_dir', default='/home/yang/study/experiment_result/logs', type=str, help='log save dir')
parser.add_argument('--is_train', default=False, type=bool, help='train?')

cfg = parser.parse_args()
images = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='x')
if __name__ == '__main__':
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:


        m4_ResNet34 = ResNet34(cfg)
        with tf.variable_scope('ResNet34'):
            m4_ResNet34.build_model(images)
        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter('{}/{}'.format(cfg.log_dir, time.strftime("%Y-%m-%d %H:%M:%S",
                                                                            time.localtime(time.time()))),
                                       sess.graph)

