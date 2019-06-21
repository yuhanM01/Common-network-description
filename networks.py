import tensorflow as tf
import numpy as np
from ops import *

class AlexNet:
    def __init__(self, cfg, keep_prob=0.8):
        '''
        :param cfg:
        :param keep_prob: dropout的参数
        '''
        self.keep_prob = keep_prob

    def build_model(self, x):
        '''
        Introduction: 搭建Vgg16网络，详见README中的Vgg16
        :param x: Input image [227, 227, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''

        # first layer
        self.conv1 = m4_conv_layers(x, 96, k_h = 11, k_w = 11, s_h = 4, s_w = 4,
                   padding = "VALID", get_vars_name=False, active_func='relu',norm=None,
                   is_trainable=self.is_train, stddev = 0.02, weight_decay=self.weight_decay, name = 'conv_1')
        self.norm1 = self.lrn(self.conv1, 2, 2e-05, 0.75, name='norm1')
        self.pool1 = m4_max_pool(self.norm1, ks=3, stride=2, padding='VALID', name='pool1')

        # second layer
        self.conv2 = m4_conv_layers(self.pool1, 256, k_h=5, k_w=5, s_h=1, s_w=1,
                                    padding="SAME", get_vars_name=False, active_func='relu', norm=None,
                                    is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay,
                                    name='conv_2')
        self.norm2 = self.lrn(self.conv2, 2, 2e-05, 0.75, name='norm2')

        self.pool2 = m4_max_pool(self.norm2, ks=3, stride=2, padding='VALID', name='pool2')

        # third layer
        self.conv3 = m4_conv_layers(self.pool2, 384, k_h=3, k_w=3, s_h=1, s_w=1,
                                    padding="SAME", get_vars_name=False, active_func='relu', norm=None,
                                    is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay,
                                    name='conv_3')

        # four layer
        self.conv4 = m4_conv_layers(self.conv3, 384, k_h=3, k_w=3, s_h=1, s_w=1,
                                    padding="SAME", get_vars_name=False, active_func='relu', norm=None,
                                    is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay,
                                    name='conv_4')

        # five layer
        self.conv5 = m4_conv_layers(self.conv4, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                    padding="SAME", get_vars_name=False, active_func='relu', norm=None,
                                    is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay,
                                    name='conv_5')
        self.pool5 = m4_max_pool(self.conv5, ks=3, stride=2, padding='VALID', name='pool5')

        batch, w, h, nc = self.pool5.get_shape().as_list()

        self.reshape = tf.reshape(self.pool5, [batch, w * h * nc])

        self.fc6 = m4_linear(self.reshape, 4096, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        self.dropput6 = self.dropout(self.fc6, self.keep_prob)

        self.fc7 = m4_linear(self.dropput6, 4096, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc7')
        self.dropout(self.fc7, self.keep_prob)

        # 接分类层


    def lrn(self, x, radius, alpha, beta, name, bias=1.0):
        """Create a local response normalization layer."""
        return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                  alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

    def dropout(self, x, keep_prob):
        """Create a dropout layer."""
        return tf.nn.dropout(x, keep_prob)





class Vgg16:
    def __init__(self, cfg):
        self.is_train = cfg.is_train #
        self.weight_decay = cfg.weight_decay # weight decay 防过拟合

    def build_model(self, x):
        '''
        Introduction: 搭建Vgg16网络，详见README中的Vgg16
        :param x: Input image [224, 224, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''
        self.conv1_1 = m4_conv_layers(x, 64, k_h = 3, k_w = 3, s_h = 1, s_w = 1,
                   padding = "SAME", get_vars_name=False, active_func='relu',norm=None,
                   is_trainable=self.is_train, stddev = 0.02, weight_decay=self.weight_decay, name = 'conv1_1')
        self.conv1_2 = m4_conv_layers(self.conv1_1, 64, k_h=3, k_w=3, s_h=1, s_w=1,
                                                  padding="SAME", get_vars_name=False,
                                                  active_func='relu', norm=None,
                                                  is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv1_2')
        self.pool1 = m4_max_pool(self.conv1_2, padding='SAME', name='pool1')

        self.conv2_1 = m4_conv_layers(self.pool1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv2_1')
        self.conv2_2 = m4_conv_layers(self.conv2_1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv2_2')
        self.pool2 = m4_max_pool(self.conv2_2, padding='SAME', name='pool2')

        self.conv3_1 = m4_conv_layers(self.pool2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv3_1')
        self.conv3_2 = m4_conv_layers(self.conv3_1, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv3_2')
        self.conv3_3 = m4_conv_layers(self.conv3_2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv3_3')
        self.pool3 = m4_max_pool(self.conv3_3, padding='SAME', name='pool3')

        self.conv4_1 = m4_conv_layers(self.pool3, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv4_1')
        self.conv4_2 = m4_conv_layers(self.conv4_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv4_2')
        self.conv4_3 = m4_conv_layers(self.conv4_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv4_3')
        self.pool4 = m4_max_pool(self.conv4_3, padding='SAME', name='pool4')

        self.conv5_1 = m4_conv_layers(self.pool4, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv5_1')
        self.conv5_2 = m4_conv_layers(self.conv5_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv5_2')
        self.conv5_3 = m4_conv_layers(self.conv5_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv5_3')
        self.pool5 = m4_max_pool(self.conv5_3, padding='SAME', name='pool5')

        batch, w, h, nc = self.pool5.get_shape().as_list()

        self.reshape = tf.reshape(self.pool5, [batch, w * h * nc])

        self.fc6 = m4_linear(self.reshape, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        self.fc7 = m4_linear(self.fc6, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc7')
        self.fc8 = m4_linear(self.fc7, 4096, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc8')
        return self.fc8

class Vgg19:

    def __init__(self, cfg):
        self.is_train = cfg.is_train
        self.weight_decay = cfg.weight_decay

    def build_model(self, x):
        '''
        Introduction: 搭建Vgg19网络，详见README中的Vgg19
        :param x: Input image [224, 224, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''
        self.conv1_1 = m4_conv_layers(x, 64, k_h = 3, k_w = 3, s_h = 1, s_w = 1,
                   padding = "SAME", get_vars_name=False, active_func='relu',norm=None,
                   is_trainable=self.is_train, stddev = 0.02, weight_decay=self.weight_decay, name = 'conv1_1')
        self.conv1_2 = m4_conv_layers(self.conv1_1, 64, k_h=3, k_w=3, s_h=1, s_w=1,
                                                  padding="SAME", get_vars_name=False,
                                                  active_func='relu', norm=None,
                                                  is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv1_2')
        self.pool1 = m4_max_pool(self.conv1_2, padding='SAME', name='pool1')

        self.conv2_1 = m4_conv_layers(self.pool1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv2_1')
        self.conv2_2 = m4_conv_layers(self.conv2_1, 128, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv2_2')
        self.pool2 = m4_max_pool(self.conv2_2, padding='SAME', name='pool2')

        self.conv3_1 = m4_conv_layers(self.pool2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv3_1')
        self.conv3_2 = m4_conv_layers(self.conv3_1, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv3_2')
        self.conv3_3_add = m4_conv_layers(self.conv3_2, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv3_3_add')
        self.conv3_3 = m4_conv_layers(self.conv3_3_add, 256, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv3_3')
        self.pool3 = m4_max_pool(self.conv3_3, padding='SAME', name='pool3')

        self.conv4_1 = m4_conv_layers(self.pool3, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv4_1')
        self.conv4_2 = m4_conv_layers(self.conv4_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv4_2')
        self.conv4_3_add = m4_conv_layers(self.conv4_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv4_3_add')
        self.conv4_3 = m4_conv_layers(self.conv4_3_add, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv4_3')
        self.pool4 = m4_max_pool(self.conv4_3, padding='SAME', name='pool4')

        self.conv5_1 = m4_conv_layers(self.pool4, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv5_1')
        self.conv5_2 = m4_conv_layers(self.conv5_1, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv5_2')
        self.conv5_3_add = m4_conv_layers(self.conv5_2, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv5_3_add')
        self.conv5_3 = m4_conv_layers(self.conv5_3_add, 512, k_h=3, k_w=3, s_h=1, s_w=1,
                                      padding="SAME", get_vars_name=False,
                                      active_func='relu', norm=None,
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv5_3')
        self.pool5 = m4_max_pool(self.conv5_3, padding='SAME', name='pool5')

        batch, w, h, nc = self.pool5.get_shape().as_list()

        self.reshape = tf.reshape(self.pool5, [batch, w * h * nc])

        self.fc6 = m4_linear(self.reshape, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        self.fc7 = m4_linear(self.fc6, 4096, active_function='relu', norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc7')
        self.fc8 = m4_linear(self.fc7, 4096, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc8')
        return self.fc8

class ResNet18:
    def __init__(self,cfg):
        self.is_train = cfg.is_train
        self.weight_decay = cfg.weight_decay

    def build_model(self, x):
        '''
        Introduction: 搭建ResNet18网络，详见README中的ResNet18
        :param x:Input image [224, 224, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''
        residual_list = [2, 2, 2, 2]
        x = m4_conv_layers(x, 64, k_h=7, k_w=7, s_h=2, s_w=2,
                                      padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv1_1')
        x = m4_max_pool(x, ks=3, stride=2, padding='SAME', name='pool1')

        for i in range(residual_list[0]):
            x = m4_resblock(x, 64, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=False,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock2_' + str(i))

        for i in range(residual_list[1]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_resblock(x, 128, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock3_' + str(i))

        for i in range(residual_list[2]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_resblock(x, 256, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock4_' + str(i))

        for i in range(residual_list[3]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_resblock(x, 512, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock5_' + str(i))

        x = m4_average_pool(x, ks=7, stride=7, padding='SAME', name='average_pool')
        _, w, h, nc = x.get_shape().as_list()
        x = tf.reshape(x, [-1, w * h * nc])
        self.fc6 = m4_linear(x, 1000, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        return self.fc6

class ResNet34:
    def __init__(self,cfg):
        self.is_train = cfg.is_train
        self.weight_decay = cfg.weight_decay

    def build_model(self, x):
        '''
        Introduction: 搭建ResNet34网络，详见README中的ResNet34
        :param x:Input image [224, 224, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''
        residual_list = [3, 4, 6, 3]
        x = m4_conv_layers(x, 64, k_h=7, k_w=7, s_h=2, s_w=2,
                                      padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv1_1')
        x = m4_max_pool(x, ks=3, stride=2, padding='SAME', name='pool1')

        for i in range(residual_list[0]):
            x = m4_resblock(x, 64, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=False,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock2_' + str(i))

        for i in range(residual_list[1]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_resblock(x, 128, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock3_' + str(i))

        for i in range(residual_list[2]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_resblock(x, 256, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock4_' + str(i))

        for i in range(residual_list[3]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_resblock(x, 512, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='resblock5_' + str(i))

        x = m4_average_pool(x, ks=7, stride=7, padding='SAME', name='average_pool')
        _, w, h, nc = x.get_shape().as_list()
        x = tf.reshape(x, [-1, w * h * nc])
        self.fc6 = m4_linear(x, 1000, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        return self.fc6


class ResNet50:
    def __init__(self,cfg):
        self.is_train = cfg.is_train
        self.weight_decay = cfg.weight_decay

    def build_model(self, x):
        '''
        Introduction: 搭建ResNet50网络，详见README中的ResNet50
        :param x:Input image [224, 224, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''
        residual_list = [3, 4, 6, 3]
        x = m4_conv_layers(x, 64, k_h=7, k_w=7, s_h=2, s_w=2,
                                      padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv1_1')
        x = m4_max_pool(x, ks=3, stride=2, padding='SAME', name='pool1')

        for i in range(residual_list[0]):
            x = m4_bottle_resblock(x, 64, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=False,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock2_' + str(i))

        for i in range(residual_list[1]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 128, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock3_' + str(i))

        for i in range(residual_list[2]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 256, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock4_' + str(i))

        for i in range(residual_list[3]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 512, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock5_' + str(i))

        x = m4_average_pool(x, ks=7, stride=7, padding='SAME', name='average_pool')
        _, w, h, nc = x.get_shape().as_list()
        x = tf.reshape(x, [-1, w * h * nc])
        self.fc6 = m4_linear(x, 1000, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        return self.fc6

class ResNet101:
    def __init__(self,cfg):
        self.is_train = cfg.is_train
        self.weight_decay = cfg.weight_decay

    def build_model(self, x):
        '''
        Introduction: 搭建ResNet101网络，详见README中的ResNet101
        :param x:Input image [224, 224, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''
        residual_list = [3, 4, 23, 3]
        x = m4_conv_layers(x, 64, k_h=7, k_w=7, s_h=2, s_w=2,
                                      padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv1_1')
        x = m4_max_pool(x, ks=3, stride=2, padding='SAME', name='pool1')

        for i in range(residual_list[0]):
            x = m4_bottle_resblock(x, 64, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=False,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock2_' + str(i))

        for i in range(residual_list[1]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 128, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock3_' + str(i))

        for i in range(residual_list[2]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 256, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock4_' + str(i))

        for i in range(residual_list[3]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 512, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock5_' + str(i))

        x = m4_average_pool(x, ks=7, stride=7, padding='SAME', name='average_pool')
        _, w, h, nc = x.get_shape().as_list()
        x = tf.reshape(x, [-1, w * h * nc])
        self.fc6 = m4_linear(x, 1000, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        return self.fc6

class ResNet152:
    def __init__(self,cfg):
        self.is_train = cfg.is_train
        self.weight_decay = cfg.weight_decay

    def build_model(self, x):
        '''
        Introduction: 搭建ResNet152网络，详见README中的ResNet152
        :param x:Input image [224, 224, 3]
        :return: softmax之前的那个全连接层，即特征映射
        '''
        residual_list = [3, 8, 36, 3]
        x = m4_conv_layers(x, 64, k_h=7, k_w=7, s_h=2, s_w=2,
                                      padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                                      is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='conv1_1')
        x = m4_max_pool(x, ks=3, stride=2, padding='SAME', name='pool1')

        for i in range(residual_list[0]):
            x = m4_bottle_resblock(x, 64, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=False,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock2_' + str(i))

        for i in range(residual_list[1]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 128, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock3_' + str(i))

        for i in range(residual_list[2]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 256, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock4_' + str(i))

        for i in range(residual_list[3]):
            if i==0:
                is_downSample = True
            else:
                is_downSample = False
            x = m4_bottle_resblock(x, 512, k_h=3, k_w=3, s_h=1, s_w=1,is_downsample=is_downSample,
                        padding="SAME", get_vars_name=False, active_func='relu', norm='batch_norm',
                        is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='bottle_resblock5_' + str(i))

        x = m4_average_pool(x, ks=7, stride=7, padding='SAME', name='average_pool')
        _, w, h, nc = x.get_shape().as_list()
        x = tf.reshape(x, [-1, w * h * nc])
        self.fc6 = m4_linear(x, 1000, active_function=None, norm=None, get_vars_name=False,
                             is_trainable=self.is_train, stddev=0.02, weight_decay=self.weight_decay, name='fc6')
        return self.fc6