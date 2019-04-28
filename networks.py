import tensorflow as tf
import numpy as np
from ops import *

class Vgg16:
    def __init__(self, cfg):
        self.is_train = cfg.is_train

    def build_model(self, x):
        self.conv1_1 = m4_conv_layers(x, 64, k_h = 3, k_w = 3, s_h = 1, s_w = 1,
                   padding = "SAME", get_vars_name=False, active_func='relu',norm=None,
                   is_trainable=self.is_train, stddev = 0.02, name = 'conv1_1')
        self.conv1_2 = m4_conv_layers(self.conv1_1, 64, k_h=3, k_w=3, s_h=1, s_w=1,
                                                  padding="SAME", get_vars_name=False,
                                                  active_func='relu', norm=None,
                                                  is_trainable=self.is_train, stddev=0.02, name='conv1_2')
        self.pool1 = m4_max_pool(self.conv1_2, padding='SAME', name='pool1')

        self.conv2_1 = m4_conv_layers(self.pool1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv2_1')
        self.conv2_2 = m4_conv_layers(self.conv2_1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv2_2')
        self.pool2 = m4_max_pool(self.conv2_2, padding='SAME', name='pool2')

        self.conv3_1 = m4_conv_layers(self.pool2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv3_1')
        self.conv3_2 = m4_conv_layers(self.conv3_1, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv3_2')
        self.conv3_3 = m4_conv_layers(self.conv3_2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv3_3')
        self.pool3 = m4_max_pool(self.conv3_3, padding='SAME', name='pool3')

        self.conv4_1 = m4_conv_layers(self.pool3, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv4_1')
        self.conv4_2 = m4_conv_layers(self.conv4_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv4_2')
        self.conv4_3 = m4_conv_layers(self.conv4_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv4_3')
        self.pool4 = m4_max_pool(self.conv4_3, padding='SAME', name='pool4')

        self.conv5_1 = m4_conv_layers(self.pool4, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv5_1')
        self.conv5_2 = m4_conv_layers(self.conv5_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv5_2')
        self.conv5_3 = m4_conv_layers(self.conv5_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv5_3')
        self.pool5 = m4_max_pool(self.conv5_3, padding='SAME', name='pool5')

        batch, w, h, nc = self.pool5.get_shape().as_list()

        self.reshape = tf.reshape(self.pool5, [batch, w * h * nc])

        self.fc6 = m4_linear(self.reshape, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, name='fc6')
        self.fc7 = m4_linear(self.fc6, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, name='fc7')
        self.fc8 = m4_linear(self.fc7, 4096, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, name='fc8')
        return self.fc8

class Vgg19:
    def __init__(self, cfg):
        self.is_train = cfg.is_train

    def build_model(self, x):
        self.conv1_1 = m4_conv_layers(x, 64, k_h = 3, k_w = 3, s_h = 1, s_w = 1,
                   padding = "SAME", get_vars_name=False, active_func='relu',norm=None,
                   is_trainable=self.is_train, stddev = 0.02, name = 'conv1_1')
        self.conv1_2 = m4_conv_layers(self.conv1_1, 64, k_h=3, k_w=3, s_h=1, s_w=1,
                                                  padding="SAME", get_vars_name=False,
                                                  active_func='relu', norm=None,
                                                  is_trainable=self.is_train, stddev=0.02, name='conv1_2')
        self.pool1 = m4_max_pool(self.conv1_2, padding='SAME', name='pool1')

        self.conv2_1 = m4_conv_layers(self.pool1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv2_1')
        self.conv2_2 = m4_conv_layers(self.conv2_1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv2_2')
        self.pool2 = m4_max_pool(self.conv2_2, padding='SAME', name='pool2')

        self.conv3_1 = m4_conv_layers(self.pool2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv3_1')
        self.conv3_2 = m4_conv_layers(self.conv3_1, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv3_2')
        self.conv3_3_add = m4_conv_layers(self.conv3_2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv3_3_add')
        self.conv3_3 = m4_conv_layers(self.conv3_3_add, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv3_3')
        self.pool3 = m4_max_pool(self.conv3_3, padding='SAME', name='pool3')

        self.conv4_1 = m4_conv_layers(self.pool3, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv4_1')
        self.conv4_2 = m4_conv_layers(self.conv4_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv4_2')
        self.conv4_3_add = m4_conv_layers(self.conv4_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv4_3_add')
        self.conv4_3 = m4_conv_layers(self.conv4_3_add, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv4_3')
        self.pool4 = m4_max_pool(self.conv4_3, padding='SAME', name='pool4')

        self.conv5_1 = m4_conv_layers(self.pool4, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv5_1')
        self.conv5_2 = m4_conv_layers(self.conv5_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv5_2')
        self.conv5_3_add = m4_conv_layers(self.conv5_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv5_3_add')
        self.conv5_3 = m4_conv_layers(self.conv5_3_add, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, name='conv5_3')
        self.pool5 = m4_max_pool(self.conv5_3, padding='SAME', name='pool5')

        batch, w, h, nc = self.pool5.get_shape().as_list()

        self.reshape = tf.reshape(self.pool5, [batch, w * h * nc])

        self.fc6 = m4_linear(self.reshape, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, name='fc6')
        self.fc7 = m4_linear(self.fc6, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, name='fc7')
        self.fc8 = m4_linear(self.fc7, 4096, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, name='fc8')
        return self.fc8

class ResNet18:
    def __init__(self,cfg):
        self.is_train = cfg.is_train

    def build_model(self, x):
        '''
        :param x:
        :return:
        '''
        residual_list = [2, 2, 2, 2]
        self.conv1_1 = m4_conv_layers(x, 64, k_h=7, k_w=7, s_h=2, s_w=2,
                                      padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                      is_trainable=self.is_train, stddev=0.02, name='conv1_1')
        self.pool1 = m4_max_pool(self.conv1_1, ks=3, stride=2, padding='SAME', name='pool1')


