
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys
import os

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from GCNet_tf import *
from tf_RRBlock import SMB, gumbel_softmax

config = tf.ConfigProto()   
config.gpu_options.allow_growth = True 
session = tf.InteractiveSession(config=config)  

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.read_file(filename)
    image = tf.image.decode_image(string, channels=3)  
    image = tf.cast(image, tf.float32)  
    image /= 255   
    return image


def quantize_image(image):
    image = tf.round(image * 255) 
    image = tf.saturate_cast(image, tf.uint8) 
    return image


def write_png(filename, image):
    """Saves an image to a PNG file."""
    image = quantize_image(image)
    string = tf.image.encode_png(image) 
    return tf.write_file(filename, string)


class AnalysisTransform(tf.keras.layers.Layer): 
    """The analysis transform."""
    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(AnalysisTransform, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_0")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_1")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="gdn_2")),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_3", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]  
        super(AnalysisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor

class SynthesisTransform(tf.keras.layers.Layer): 
    """The synthesis transform."""

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super(SynthesisTransform, self).__init__(*args, **kwargs)   

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_0", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True)),
            tfc.SignalConv2D(
                self.num_filters, (5, 5), name="layer_2", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True)),
            tfc.SignalConv2D(
                3, (5, 5), name="layer_3", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]   
        super(SynthesisTransform, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor

class IEB_head(tf.keras.layers.Layer):
    '''
    Information Enhance Block
    input:[n,h,w,c]
    output:[n,2h,2w,c/2] 
    '''
    def __init__(self, num_filters, gcb, *args, **kwargs):
        self.num_filters = num_filters
        self.gcb = gcb
        super(IEB_head, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters//2, (3, 3), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters//2, (1, 1), name="layer_1", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(IEB_head, self).build(input_shape)

    def call(self, tensor_1):
        x1 = self._layers[0](tensor_1)  
        x2 = global_context_block(x1, channels=self.num_filters//2, sn=True, scope=self.gcb) 
        x4 = self._layers[1](x2)     
        return x4

class IEB_body(tf.keras.layers.Layer):
    '''
    Information Enhance Block
    input:[n,h,w,c],[n,2h,2w,c/2]
    output:[n,2h,2w,c/2] 
    '''
    def __init__(self, num_filters, gcb, *args, **kwargs):
        self.num_filters = num_filters
        self.gcb = gcb
        super(IEB_body, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters//2, (3, 3), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters//2, (1, 1), name="layer_1", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(IEB_body, self).build(input_shape)

    def call(self, tensor_1, tnesor_2):
        x1 = self._layers[0](tensor_1) 
        x2 = global_context_block(x1, channels=self.num_filters//2, sn=True, scope=self.gcb) 
        x3 = tf.concat([x2, tnesor_2], -1)  
        x4 = self._layers[1](x3) 
        return x4 

class IEB_tail(tf.keras.layers.Layer):
    '''
    Information Enhance Block
    input:[n,h,w,c],[n,2h,2w,c/2]
    output:[n,2h,2w,c] 
    '''
    def __init__(self, num_filters, gcb, *args, **kwargs):
        self.num_filters = num_filters
        self.gcb = gcb
        super(IEB_tail, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters//2, (3, 3), name="layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu),
            tfc.SignalConv2D(
                self.num_filters, (1, 1), name="layer_1", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True,
                activation=None),
        ]
        super(IEB_tail, self).build(input_shape)

    def call(self, tensor_1, tnesor_2):
        x1 = self._layers[0](tensor_1)
        x2 = global_context_block(x1, channels=self.num_filters//2, sn=True, scope=self.gcb)
        x3 = tf.concat([x2, tnesor_2], -1)
        x4 = self._layers[1](x3)
        return x4

class RRB_head(tf.keras.layers.Layer):
    '''
    Redundant Remove Block
    input:[n,2h,2w,c] 
    output:[n,h,w,c]
    '''
    def __init__(self, channels, nm, *args, **kwargs):
        self.channels = channels
        self.nm = nm
        super(RRB_head, self).__init__(*args, **kwargs)
    
    def build(self, input_shape):

        self.down = tfc.SignalConv2D(
                self.channels, (3, 3), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu)

        # spatial mask
        self.spa_mask = [
            tf.keras.layers.Conv2D(self.channels//4, kernel_size=3, strides=(1, 1), padding="same",
                                    name = self.nm+'_conv0', activation=tf.nn.relu),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(self.channels//4, kernel_size=3, strides=(1, 1), padding="same",
                                    name = self.nm+'_conv1', activation=tf.nn.relu),
            tf.keras.layers.Conv2D(self.channels//4, kernel_size=3, strides=(1, 1), padding="same",
                                    name = self.nm+'_conv2', activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(2, kernel_size=3, strides=(2, 2), padding="valid",
                                    name = self.nm+'_conv0tran', activation=tf.nn.relu),
        ]

        # body
        self.body = SMB(self.channels, self.channels, n_layers=3)

        self.tau = 1

        self.c2c = [tfc.SignalConv2D(
                self.channels, (1, 1), name="layer_1", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True,
                activation=None),]
        
        super(RRB_head, self).build(input_shape)

    def _update_tau(self, tau):
        self.tau = tau

    def call(self, tensor):
        tensor = self.down(tensor)

        tem = self.spa_mask[0](tensor)
        tem = self.spa_mask[1](tem)
        tem = self.spa_mask[2](tem)
        tem = self.spa_mask[3](tem)
        spa_mask = self.spa_mask[4](tem)
        spa_mask = gumbel_softmax(spa_mask, 3, self.tau)

        # tmp = x * spa_mask[:, 1:, 1:, ...]
        out0, ch_mask = self.body([tensor, spa_mask[:, 1:, 1:, 1:]])

        # out = out + x
        out1 = global_context_block(out0, channels=self.channels, sn=True, scope=self.nm) + tensor
        out = self.c2c[0](out1)
        return out

class RRB_body(tf.keras.layers.Layer):
    '''
    Redundant Remove Block
    input:[n,2h,2w,c/2],[n,h,w,c]
    output:[n,h,w,c]
    '''
    def __init__(self, channels, nm, *args, **kwargs):
        self.channels = channels
        self.nm = nm
        super(RRB_body, self).__init__(*args, **kwargs)
    
    def build(self, input_shape):

        self.down = tfc.SignalConv2D(
                self.channels, (3, 3), name="layer_1", corr=True, strides_down=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.relu)

        # spatial mask
        self.spa_mask = [
            tf.keras.layers.Conv2D(self.channels//4, kernel_size=3, strides=(1, 1), padding="same",
                                    name = self.nm+'_conv0', activation=tf.nn.relu),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(self.channels//4, kernel_size=3, strides=(1, 1), padding="same",
                                    name = self.nm+'_conv1', activation=tf.nn.relu),
            tf.keras.layers.Conv2D(self.channels//4, kernel_size=3, strides=(1, 1), padding="same",
                                    name = self.nm+'_conv2', activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(2, kernel_size=3, strides=(2, 2), padding="valid",
                                    name = self.nm+'_conv0tran', activation=tf.nn.relu),
        ]

        # body
        self.body = SMB(self.channels, self.channels, n_layers=3)
        self.tau = 1
        self.c2c = [tfc.SignalConv2D(
                self.channels, (1, 1), name="layer_1", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True,
                activation=None),]
        
        super(RRB_body, self).build(input_shape)

    def _update_tau(self, tau):
        self.tau = tau

    def call(self, x, y):
        tensor = self.down(x)

        tem = self.spa_mask[0](tensor)
        tem = self.spa_mask[1](tem)
        tem = self.spa_mask[2](tem)
        tem = self.spa_mask[3](tem)
        spa_mask = self.spa_mask[4](tem)
        spa_mask = gumbel_softmax(spa_mask, 3, self.tau)

        out0, ch_mask = self.body([tensor, spa_mask[:, 1:, 1:, 1:]])
        out1 = global_context_block(out0, channels=self.channels, sn=True, scope=self.nm) + tensor

        con = tf.concat([out1, y], -1)
        out = self.c2c[0](con)

        return out

class PredictionTransform_spatial(tf.keras.layers.Layer): 
    # input:[n,h,w,c] output:[n,2h,2w,c]
    def __init__(self, num_filters_first, num_filters_last, nm, *args, **kwargs):
        self.num_filters_first = num_filters_first
        self.num_filters_last = num_filters_last
        self.nm = nm
        super(PredictionTransform_spatial, self).__init__(*args, **kwargs)

    def build(self, input_shape):
    
        self.IEB = [
            IEB_head(self.num_filters_first, 'IEB1'+self.nm),
            IEB_body(self.num_filters_first, 'IEB2'+self.nm),
            IEB_tail(self.num_filters_last, 'IEB3'+self.nm),
        ]

        self.RRB = [
            RRB_body(self.num_filters_first, 'RRB1'+self.nm),
            RRB_body(self.num_filters_last, 'RRB2'+self.nm),
        ]

        super(PredictionTransform_spatial, self).build(input_shape)

    def call(self, tensor):
        x1 = self.IEB[0](tensor)
        x2 = self.RRB[0](x1, tensor)
        x3 = self.IEB[1](x2,x1)
        x4 = self.RRB[1](x3, x2)
        x5 = self.IEB[2](x4,x3)
        return x5

class Uncertainty_spatial(tf.keras.layers.Layer):
    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        self.n_colors = 3
        super(Uncertainty_spatial, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self._layers = [
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="Uncertaint_layer_0", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=None),
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="Uncertaint_layer_1", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.elu),
            tfc.SignalConv2D(
                self.num_filters, (3, 3), name="Uncertaint_layer_2", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.elu),
            tfc.SignalConv2D(
                self.n_colors, (3, 3), name="Uncertaint_layer_3", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.elu),
        ]
        super(Uncertainty_spatial, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.io.parse_single_example(serialized_example,
                                       features={
                                           'img_raw_org': tf.io.FixedLenFeature([], tf.string),
                                           'img_raw_half': tf.io.FixedLenFeature([], tf.string),
                                           'img_raw_quater': tf.io.FixedLenFeature([], tf.string),
                                       })

    img_org = tf.decode_raw(features['img_raw_org'], tf.uint8)
    img_half = tf.decode_raw(features['img_raw_half'], tf.uint8)
    img_quater = tf.decode_raw(features['img_raw_quater'], tf.uint8)

    img_org = tf.reshape(img_org, [512, 512, 3])
    img_half = tf.reshape(img_half, [256, 256, 3])
    img_quater = tf.reshape(img_quater, [128, 128, 3])

    img_org = tf.cast(img_org, tf.float32) * (1. / 255)
    img_half = tf.cast(img_half, tf.float32) * (1. / 255)
    img_quater = tf.cast(img_quater, tf.float32) * (1. / 255)
    return img_org, img_half, img_quater


def train(args,iter):
    tf.compat.v1.reset_default_graph()
    if args.verbose:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


    img_org, img_half, img_quater = read_and_decode(args.train_tfrecords)
    img_org_batch, img_half_batch, img_quater_batch = tf.train.shuffle_batch([img_org, img_half, img_quater],
                                                      batch_size=args.batchsize,
                                                      capacity=120,
                                                      min_after_dequeue=50)
    num_pixels_B = args.batchsize * args.patchsize ** 2 /16
    num_pixels_e1 = args.batchsize * args.patchsize ** 2 /4
    num_pixels_e2 = args.batchsize * args.patchsize ** 2

    inputx = tf.compat.v1.placeholder(dtype=tf.float32, shape=[args.batchsize, args.patchsize, args.patchsize, 3], name='input_org')
    x = inputx
    x_half = tf.compat.v1.placeholder(dtype=tf.float32, shape=[args.batchsize, args.patchsize_h, args.patchsize_h, 3], name='input_half')
    x_quater = tf.compat.v1.placeholder(dtype=tf.float32, shape=[args.batchsize, args.patchsize_q, args.patchsize_q, 3], name='input_quater')


    with tf.variable_scope('Compress'):
        analysis_transform_B = AnalysisTransform(args.num_filters_B)
        analysis_transform_e1 = AnalysisTransform(args.num_filters_e1)
        analysis_transform_e2 = AnalysisTransform(args.num_filters_e2)
        synthesis_transform_B = SynthesisTransform(args.num_filters_B)
        synthesis_transform_e1 = SynthesisTransform(args.num_filters_e1)
        synthesis_transform_e2 = SynthesisTransform(args.num_filters_e2)
        entropy_bottleneck_B = tfc.EntropyBottleneck()
        entropy_bottleneck_e1 = tfc.EntropyBottleneck()
        entropy_bottleneck_e2 = tfc.EntropyBottleneck()
        prediction_e1 = PredictionTransform_spatial(args.num_filters_B,
                                            args.num_filters_e1, 'a')
        prediction_e2 = PredictionTransform_spatial(args.num_filters_e1,
                                            args.num_filters_e2, 'b')

        y_B = analysis_transform_B(x_quater)
        y_B_tilde, y_B_likelihoods = entropy_bottleneck_B(y_B, training=True)

        y_e1 = analysis_transform_e1(x_half)
        y_e1_predict = prediction_e1(y_B_tilde)
        y_e1_res_tilde, y_e1_res_likelihoods = entropy_bottleneck_e1(tf.subtract(y_e1, y_e1_predict), training=True)
        y_e1_tilde = y_e1_predict + y_e1_res_tilde


        y_e2 = analysis_transform_e2(x)
        y_e2_predict = prediction_e2(y_e1_tilde)
        y_e2_res_tilde, y_e2_res_likelihoods = entropy_bottleneck_e2(tf.subtract(y_e2, y_e2_predict), training=True)
        y_e2_tilde = y_e2_predict + y_e2_res_tilde

        x_B_hat = synthesis_transform_B(y_B_tilde)
        x_e1_hat = synthesis_transform_e1(y_e1_tilde)
        x_e2_hat = synthesis_transform_e2(y_e2_tilde)


    with tf.variable_scope('Uncertainty'):
        uncertainty_e1 = Uncertainty_spatial(args.num_filters_B)
        uncertainty_e2 = Uncertainty_spatial(args.num_filters_e1)
    
        var_b = uncertainty_e1(y_e1_tilde)
        var_e1 = uncertainty_e2(y_e2_tilde)


    
    stage1_s_b = tf.exp(-2 * var_b)
    stage1_x_quater_ = tf.multiply(x_quater, stage1_s_b)
    stage1_x_B_hat_ = tf.multiply(x_B_hat, stage1_s_b)
    stage1_uncer_mse_B = tf.reduce_mean(tf.squared_difference(stage1_x_quater_, stage1_x_B_hat_))
    stage1_uncer_B = tf.reduce_mean(2 * var_b)
    stage1_train_mse_B = stage1_uncer_mse_B + 5 * stage1_uncer_B
    
    stage1_s_e1 = tf.exp(-2 * var_e1)
    stage1_x_half_ = tf.multiply(x_half, stage1_s_e1)
    stage1_x_e1_hat_ = tf.multiply(x_e1_hat, stage1_s_e1)
    stage1_uncer_mse_e1 = tf.reduce_mean(tf.squared_difference(stage1_x_half_, stage1_x_e1_hat_))
    stage1_uncer_e1 = tf.reduce_mean(2 * var_e1)
    stage1_train_mse_e1 = stage1_uncer_mse_e1 + 5 * stage1_uncer_e1


    b, h, w, c = var_b.get_shape().as_list()
    stage2_s1_b = tf.reshape(var_b, [b, -1, c])
    stage2_pmin_b = tf.reduce_min(stage2_s1_b, axis=1)
    stage2_pmin_b = tf.expand_dims(tf.expand_dims(stage2_pmin_b, dim=1),dim=1)
    stage2_s_b = var_b
    stage2_s_b = stage2_s_b - stage2_pmin_b + 1
    stage2_x_quater_ = tf.multiply(x_quater, stage2_s_b)
    stage2_x_B_hat_ = tf.multiply(x_B_hat, stage2_s_b)
    stage2_uncer_mse_B = tf.reduce_mean(tf.squared_difference(stage2_x_quater_, stage2_x_B_hat_))
    stage2_train_mse_B = stage2_uncer_mse_B

    b, h, w, c = var_e1.get_shape().as_list()
    stage2_s1_e1 = tf.reshape(var_e1, [b, -1, c])
    stage2_pmin_e1 = tf.reduce_min(stage2_s1_e1, axis=1)
    stage2_pmin_e1 = tf.expand_dims(tf.expand_dims(stage2_pmin_e1, dim=1),dim=1)
    stage2_s_e1 = var_e1
    stage2_s_e1 = stage2_s_e1 - stage2_pmin_e1 + 1
    stage2_x_half_ = tf.multiply(x_half, stage2_s_e1)
    stage2_x_e1_hat_ = tf.multiply(x_e1_hat, stage2_s_e1)
    stage2_uncer_mse_e1 = tf.reduce_mean(tf.squared_difference(stage2_x_half_, stage2_x_e1_hat_))
    stage2_train_mse_e1 = stage2_uncer_mse_e1

    

    train_bpp_B = (tf.reduce_sum(tf.log(y_B_likelihoods))) / (-np.log(2) * num_pixels_B)
    train_bpp_e1 = (tf.reduce_sum(tf.log(y_e1_res_likelihoods))) / (-np.log(2) * num_pixels_e1)
    train_bpp_e2 = (tf.reduce_sum(tf.log(y_e2_res_likelihoods))) / (-np.log(2) * num_pixels_e2)

    train_mse_B = tf.reduce_mean(tf.squared_difference(x_quater, x_B_hat))
    train_mse_e1 = tf.reduce_mean(tf.squared_difference(x_half, x_e1_hat))
    train_mse_e2 = tf.reduce_mean(tf.squared_difference(x, x_e2_hat))

    train_mse_B *= 255 ** 2
    train_mse_e1 *= 255 ** 2
    train_mse_e2 *= 255 ** 2


    stage1_train_mse_B *= 255 ** 2
    stage1_train_mse_e1 *= 255 ** 2
    stage2_train_mse_B *= 255 ** 2
    stage2_train_mse_e1 *= 255 ** 2

    

    stage1_train_loss_B = args.lmbdaB * train_bpp_B + stage1_train_mse_B
    stage1_train_loss_e1 = args.lmbda1 * train_bpp_e1 + stage1_train_mse_e1

    stage2_train_loss_B = args.lmbdaB * train_bpp_B + stage2_train_mse_B
    stage2_train_loss_e1 = args.lmbda1 * train_bpp_e1 + stage2_train_mse_e1

    train_loss_B = args.lmbdaB * train_bpp_B + train_mse_B
    train_loss_e1 = args.lmbda1 * train_bpp_e1 + train_mse_e1
    train_loss_e2 = args.lmbda2 * train_bpp_e2 + train_mse_e2
    
    stage0_train_loss = train_loss_B + train_loss_e1 + train_loss_e2
    stage1_train_loss = stage1_train_loss_B + stage1_train_loss_e1 + train_loss_e2
    stage2_train_loss = stage2_train_loss_B + stage2_train_loss_e1 + train_loss_e2


    step = tf.train.create_global_step()

    aux_optimizer_B = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step_B = aux_optimizer_B.minimize(entropy_bottleneck_B.losses[0])
    aux_optimizer_e1 = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step_e1 = aux_optimizer_e1.minimize(entropy_bottleneck_e1.losses[0])
    aux_optimizer_e2 = tf.train.AdamOptimizer(learning_rate=1e-3)
    aux_step_e2 = aux_optimizer_e2.minimize(entropy_bottleneck_e2.losses[0])

    code_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Compress')
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Uncertainty|Compress')

    train_stage0 = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(stage0_train_loss, global_step=step, var_list = code_var_list)
    train_stage1 = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(stage1_train_loss, global_step=step, var_list = var_list)
    train_stage2 = tf.train.AdamOptimizer(learning_rate=2e-5).minimize(stage2_train_loss, global_step=step, var_list = code_var_list)


    

    train_op_step0 = tf.group(train_stage0, aux_step_B, entropy_bottleneck_B.updates[0], aux_step_e1, entropy_bottleneck_e1.updates[0],
                        aux_step_e2, entropy_bottleneck_e2.updates[0])

    train_op_step1 = tf.group(train_stage1, aux_step_B, entropy_bottleneck_B.updates[0], aux_step_e1, entropy_bottleneck_e1.updates[0],
                        aux_step_e2, entropy_bottleneck_e2.updates[0])

    train_op_step2 = tf.group(train_stage2, aux_step_B, entropy_bottleneck_B.updates[0], aux_step_e1, entropy_bottleneck_e1.updates[0],
                        aux_step_e2, entropy_bottleneck_e2.updates[0])



    tf.summary.scalar("loss", stage0_train_loss)
    tf.summary.scalar("lossB", train_loss_B)
    tf.summary.scalar("loss1", train_loss_e1)
    tf.summary.scalar("loss2", train_loss_e2)
    
    tf.summary.scalar("mseB", train_mse_B)
    tf.summary.scalar("mse1", train_mse_e1)
    tf.summary.scalar("mse2", train_mse_e2)

    tf.summary.scalar("bppB", train_bpp_B)
    tf.summary.scalar("bpp1", train_bpp_e1-train_bpp_B)
    tf.summary.scalar("bpp2", train_bpp_e2-train_bpp_e1)

    tf.summary.scalar("stage0_train_loss", stage0_train_loss)
    tf.summary.scalar("stage1_train_loss", stage1_train_loss)
    tf.summary.scalar("stage2_train_loss", stage2_train_loss)


    tf.summary.image("original", quantize_image(x))
    tf.summary.image("reconstrucionB", quantize_image(x_B_hat))
    tf.summary.image("reconstrucion1", quantize_image(x_e1_hat))
    tf.summary.image("reconstrucion2", quantize_image(x_e2_hat))

    hooks = {
        tf.train.StopAtStepHook(last_step=args.last_step),
        tf.train.NanTensorHook(stage0_train_loss),
        tf.train.NanTensorHook(stage1_train_loss),
        tf.train.NanTensorHook(stage2_train_loss),
    }


    with tf.train.MonitoredTrainingSession(
            hooks=hooks, checkpoint_dir=args.checkpoint_dir,
            save_checkpoint_secs=300, save_summaries_secs=180) as sess:

        x_org_ph= np.zeros(shape=[args.batchsize, args.patchsize, args.patchsize, 3], dtype=np.float32)
        x_half_ph = np.zeros(shape=[args.batchsize, args.patchsize_h, args.patchsize_h, 3], dtype=np.float32)
        x_quater_ph = np.zeros(shape=[args.batchsize, args.patchsize_q, args.patchsize_q, 3], dtype=np.float32)
        

        if iter <= 300:      
            train_op = train_op_step0
        elif iter <= 600:     
            train_op = train_op_step1
        elif iter <= 900:     
            train_op = train_op_step2
        else:
            train_op = train_op_step0
        iter = iter + 1

       
        while not sess.should_stop():
            real_org, real_half, real_quater = sess.run([img_org_batch, img_half_batch, img_quater_batch],
                                        feed_dict={inputx: x_org_ph, x_half: x_half_ph, x_quater: x_quater_ph})
            
            sess.run(train_op, feed_dict={inputx: real_org, x_half: real_half, x_quater: real_quater})
            

if __name__ == "__main__":
    parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_tfrecords', type=str, default='./clic-train-spatial.tfrecords', help="Training image path.")
    parser.add_argument('--checkpoint_dir', type=str, default='./model_spatial/', help="Checkpoint folder")
    parser.add_argument("--num_filters_B", type=int, default=96, help="Number of base layer.")
    parser.add_argument("--num_filters_e1", type=int, default=96, help="Number of filters of enhancement layer e1.")
    parser.add_argument("--num_filters_e2", type=int, default=96, help="Number of filters of enhancement layer e2.")
    parser.add_argument("--batchsize", type=int, default=8, help="Batch size.")
    parser.add_argument("--patchsize", type=int, default=512, help="Training image size.")
    parser.add_argument("--patchsize_h", type=int, default=256, help="Training image size.")
    parser.add_argument("--patchsize_q", type=int, default=128, help="Training image size.")
    parser.add_argument("--lmbdaB", type=int, default=500, help="RD trade-off of base layer.")
    parser.add_argument("--lmbda1", type=int, default=300, help="RD trade-off of enhancement layer e1.")
    parser.add_argument("--lmbda2", type=int, default=150, help="RD trade-off of enhancement layer e2.")
    parser.add_argument("--last_step", type=int, default=100000, help="Training iterations.")
    parser.add_argument("--preprocess_threads", type=int, default=4, help="Parallel preprocess threads.")
    parser.add_argument("--verbose", type=bool, default=True, help="set_verbosity.")

    args = parser.parse_args()
    
    iter = 0
    train(args, iter)
