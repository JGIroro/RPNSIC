
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
from matplotlib import pyplot
import tensorflow_compression as tfc

from GCNet_tf import *
from tf_RRBlock import SMB, gumbel_softmax
import cv2
import time

config = tf.ConfigProto()   # 配置Session的运算方式 类似to.Device
# config.gpu_options.per_process_gpu_memory_fraction=0.5
config.gpu_options.allow_growth = True  # 使用GPU
session = tf.InteractiveSession(config=config)  # 使得在启动session后还能够定义操作operation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'

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

class PredictionTransform_quality(tf.keras.layers.Layer):
    def __init__(self, num_filters_first, num_filters_last, nm, *args, **kwargs):
        self.num_filters_first = num_filters_first
        self.num_filters_last = num_filters_last
        self.nm = nm
        super(PredictionTransform_quality, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        
        self.IEB = [
            IEB_body(self.num_filters_first, 'IEB1'+self.nm),
            IEB_body(self.num_filters_first, 'IEB2'+self.nm),
            IEB_tail(self.num_filters_last, 'IEB3'+self.nm),
        ]

        self.RRB = [
            RRB_head(self.num_filters_first, 'RRB0'+self.nm),
            RRB_body(self.num_filters_first, 'RRB1'+self.nm),
            RRB_body(self.num_filters_last, 'RRB2'+self.nm),
        ]

        super(PredictionTransform_quality, self).build(input_shape)

    def call(self, tensor):
        x0 = self.RRB[0](tensor)
        x1 = self.IEB[0](x0, tensor)
        x2 = self.RRB[1](x1, x0)
        x3 = self.IEB[1](x2, x1)
        x4 = self.RRB[2](x3, x2)
        x5 = self.IEB[2](x4, x3)
        return x5
    
class Uncertainty_quality(tf.keras.layers.Layer):
    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        self.n_colors = 3
        super(Uncertainty_quality, self).__init__(*args, **kwargs)

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
                self.num_filters, (3, 3), name="Uncertaint_layer_3", corr=False, strides_up=2,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.elu),
            tfc.SignalConv2D(
                self.n_colors, (3, 3), name="Uncertaint_layer_5", corr=False, strides_up=1,
                padding="same_zeros", use_bias=True,
                activation=tf.nn.elu),
        ]
        super(Uncertainty_quality, self).build(input_shape)

    def call(self, tensor):
        for layer in self._layers:
            tensor = layer(tensor)
        return tensor


  
def evaluate(args):

    tf.reset_default_graph()

    x = read_png(args.input_image)
    x = tf.expand_dims(x, 0)
    x.set_shape([1, None, None, 3])
    x_shape = tf.shape(x)


# 实例化网络模块
    with tf.variable_scope('Compress'):
        analysis_transform_B = AnalysisTransform(args.num_filters_B)
        analysis_transform_e1 = AnalysisTransform(args.num_filters_e1)
        analysis_transform_e2 = AnalysisTransform(args.num_filters_e2)
        analysis_transform_e3 = AnalysisTransform(args.num_filters_e3)
        synthesis_transform_B = SynthesisTransform(args.num_filters_B)
        synthesis_transform_e1 = SynthesisTransform(args.num_filters_e1)
        synthesis_transform_e2 = SynthesisTransform(args.num_filters_e2)
        synthesis_transform_e3 = SynthesisTransform(args.num_filters_e3)
        entropy_bottleneck_B = tfc.EntropyBottleneck()
        entropy_bottleneck_e1 = tfc.EntropyBottleneck()
        entropy_bottleneck_e2 = tfc.EntropyBottleneck()
        entropy_bottleneck_e3 = tfc.EntropyBottleneck()
        prediction_e1 = PredictionTransform_quality(args.num_filters_B,
                                            args.num_filters_e1, 'a')
        prediction_e2 = PredictionTransform_quality(args.num_filters_e1,
                                            args.num_filters_e2, 'b')
        prediction_e3 = PredictionTransform_quality(args.num_filters_e2,
                                            args.num_filters_e3, 'c')
        
        y_B = analysis_transform_B(x)
        y_B_tilde, y_B_likelihoods = entropy_bottleneck_B(y_B, training=False)

        y_e1 = analysis_transform_e1(x)
        y_e1_predict = prediction_e1(y_B_tilde)
        y_e1_res = tf.subtract(y_e1, y_e1_predict)
        y_e1_res_tilde, y_e1_res_likelihoods = entropy_bottleneck_e1(y_e1_res, training=False)
        y_e1_tilde = y_e1_predict + y_e1_res_tilde
        y_e2_predict = prediction_e2(y_e1_tilde) 

        y_e2 = analysis_transform_e2(x)
        y_e2_res = tf.subtract(y_e2, y_e2_predict)
        y_e2_res_tilde, y_e2_res_likelihoods = entropy_bottleneck_e2(y_e2_res, training=False)
        y_e2_tilde = y_e2_predict + y_e2_res_tilde
        y_e3_predict = prediction_e3(y_e2_tilde)

        y_e3 = analysis_transform_e3(x)
        y_e3_res = tf.subtract(y_e3, y_e3_predict)
        y_e3_res_tilde, y_e3_res_likelihoods = entropy_bottleneck_e3(y_e3_res, training=False)
        y_e3_tilde = y_e3_predict + y_e3_res_tilde


        x_B_hat = synthesis_transform_B(y_B_tilde)
        x_e1_hat = synthesis_transform_e1(tf.concat([y_B_tilde, y_e1_tilde], -1))
        x_e2_hat = synthesis_transform_e2(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde], -1))
        x_e3_hat = synthesis_transform_e3(tf.concat([y_B_tilde, y_e1_tilde, y_e2_tilde, y_e3_tilde], -1))

    string_B = entropy_bottleneck_B.compress(y_B)
    string_e1 = entropy_bottleneck_e1.compress(y_e1_res)
    string_e2 = entropy_bottleneck_e2.compress(y_e2_res)
    string_e3 = entropy_bottleneck_e3.compress(y_e3_res)
    
    x_B_hat = x_B_hat[:, :x_shape[1], :x_shape[2], :]
    x_e1_hat = x_e1_hat[:, :x_shape[1], :x_shape[2], :]
    x_e2_hat = x_e2_hat[:, :x_shape[1], :x_shape[2], :]
    x_e3_hat = x_e3_hat[:, :x_shape[1], :x_shape[2], :]

    num_pixels = tf.cast(tf.reduce_prod(tf.shape(x)[:-1]), dtype=tf.float32)

    eval_bpp_B = tf.reduce_sum(tf.log(y_B_likelihoods)) / (-np.log(2) * num_pixels)
    eval_bpp_e1 = tf.reduce_sum(tf.log(y_e1_res_likelihoods)) / (-np.log(2) * num_pixels) + eval_bpp_B
    eval_bpp_e2 = tf.reduce_sum(tf.log(y_e2_res_likelihoods)) / (-np.log(2) * num_pixels) + eval_bpp_e1
    eval_bpp_e3 = tf.reduce_sum(tf.log(y_e3_res_likelihoods)) / (-np.log(2) * num_pixels) + eval_bpp_e2

    x *= 255
    x_B_hat = tf.clip_by_value(x_B_hat, 0, 1)
    x_B_hat = tf.round(x_B_hat * 255)
    x_e1_hat = tf.clip_by_value(x_e1_hat, 0, 1)
    x_e1_hat = tf.round(x_e1_hat * 255)
    x_e2_hat = tf.clip_by_value(x_e2_hat, 0, 1)
    x_e2_hat = tf.round(x_e2_hat * 255)
    x_e3_hat = tf.clip_by_value(x_e3_hat, 0, 1)
    x_e3_hat = tf.round(x_e3_hat * 255)

    mse_B = tf.reduce_mean(tf.squared_difference(x, x_B_hat))
    mse_e1 = tf.reduce_mean(tf.squared_difference(x, x_e1_hat))
    mse_e2 = tf.reduce_mean(tf.squared_difference(x, x_e2_hat))
    mse_e3 = tf.reduce_mean(tf.squared_difference(x, x_e3_hat))

    psnr_B = tf.squeeze(tf.image.psnr(x_B_hat, x, 255))
    psnr_e1 = tf.squeeze(tf.image.psnr(x_e1_hat, x, 255))
    psnr_e2 = tf.squeeze(tf.image.psnr(x_e2_hat, x, 255))
    psnr_e3 = tf.squeeze(tf.image.psnr(x_e3_hat, x, 255))

    msssim_B = tf.squeeze(tf.image.ssim_multiscale(x_B_hat, x, 255))
    msssim_e1 = tf.squeeze(tf.image.ssim_multiscale(x_e1_hat, x, 255))
    msssim_e2 = tf.squeeze(tf.image.ssim_multiscale(x_e2_hat, x, 255))
    msssim_e3 = tf.squeeze(tf.image.ssim_multiscale(x_e3_hat, x, 255))
        

    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)

        tensors_B = [string_B, tf.shape(x)[1:-1], tf.shape(y_B)[1:-1]]
        tensors_e1 = [string_e1, tf.shape(x)[1:-1], tf.shape(y_e1)[1:-1]]
        tensors_e2 = [string_e2, tf.shape(x)[1:-1], tf.shape(y_e2)[1:-1]]
        tensors_e3 = [string_e3, tf.shape(x)[1:-1], tf.shape(y_e3)[1:-1]]
        
        arrays_B = sess.run(tensors_B)
        arrays_e1 = sess.run(tensors_e1)
        arrays_e2 = sess.run(tensors_e2)
        arrays_e3 = sess.run(tensors_e3)

        packed_B = tfc.PackedTensors()
        packed_B.pack(tensors_B, arrays_B)
        with open(args.output_folder + '/stream_B.tfci', "wb") as f:
            f.write(packed_B.string)

        packed_e1 = tfc.PackedTensors()
        packed_e1.pack(tensors_e1, arrays_e1)
        with open(args.output_folder + '/stream_e1.tfci', "wb") as f:
            f.write(packed_e1.string)

        packed_e2 = tfc.PackedTensors()
        packed_e2.pack(tensors_e2, arrays_e2)
        with open(args.output_folder + '/stream_e2.tfci', "wb") as f:
            f.write(packed_e2.string)

        packed_e3 = tfc.PackedTensors()
        packed_e3.pack(tensors_e3, arrays_e3)
        with open(args.output_folder + '/stream_e3.tfci', "wb") as f:
            f.write(packed_e3.string)


        eval_bpp_B, eval_bpp_e1, eval_bpp_e2, eval_bpp_e3= sess.run(
            [eval_bpp_B, eval_bpp_e1, eval_bpp_e2, eval_bpp_e3]
        )

        mse_B, mse_e1, mse_e2, mse_e3 = sess.run(
            [mse_B, mse_e1, mse_e2, mse_e3]
        )

        psnr_B, psnr_e1, psnr_e2, psnr_e3 = sess.run(
            [psnr_B, psnr_e1, psnr_e2, psnr_e3]
        )

        msssim_B, msssim_e1, msssim_e2, msssim_e3 = sess.run(
            [msssim_B, msssim_e1, msssim_e2, msssim_e3]
        )

        num_pixels = sess.run(num_pixels)

        bpp_B = len(packed_B.string) *8 / num_pixels
        bpp_e1 = len(packed_e1.string) *8 / num_pixels
        bpp_e2 = len(packed_e2.string) *8 / num_pixels
        bpp_e3 = len(packed_e3.string) *8 / num_pixels

        ebpp_list = [eval_bpp_B, eval_bpp_e1, eval_bpp_e2, eval_bpp_e3]
        ebpp_list = [round(i,4) for i in ebpp_list]
        bpp_list = [bpp_B, bpp_e1, bpp_e2, bpp_e3]
        bpp_list = [round(i,4) for i in bpp_list]
        mse_list = [mse_B, mse_e1, mse_e2, mse_e3]
        mse_list = [round(i,4) for i in mse_list]
        psnr_list = [psnr_B, psnr_e1, psnr_e2, psnr_e3]
        psnr_list = [round(i,4) for i in psnr_list]
        msssim_list = [msssim_B, msssim_e1, msssim_e2, msssim_e3]
        msssim_list = [round(i,4) for i in msssim_list]
        
        print('eval_bpp:')
        print(ebpp_list)
        print('actual_bpp:')
        print(bpp_list)
        print('psnr:')
        print(psnr_list)
        print('msssim:')
        print(msssim_list)
        


def decompress(args):

    tf.reset_default_graph()
    
    string_B = tf.placeholder(tf.string, [1])
    string_e1 = tf.placeholder(tf.string, [1])
    string_e2 = tf.placeholder(tf.string, [1])
    string_e3= tf.placeholder(tf.string, [1])

    x_B_shape = tf.placeholder(tf.int32, [2])
    x_e1_shape = tf.placeholder(tf.int32, [2])
    x_e2_shape = tf.placeholder(tf.int32, [2])
    x_e3_shape = tf.placeholder(tf.int32, [2])

    y_B_shape = tf.placeholder(tf.int32, [2])
    y_e1_shape = tf.placeholder(tf.int32, [2])
    y_e2_shape = tf.placeholder(tf.int32, [2])
    y_e3_shape = tf.placeholder(tf.int32, [2])


    with open(args.output_folder + '/stream_B.tfci', "rb") as f:
        packed_B = tfc.PackedTensors(f.read())
    tensors_B = [string_B, x_B_shape, y_B_shape]
    arrays_B = packed_B.unpack(tensors_B)
    with open(args.output_folder + '/stream_e1.tfci', "rb") as f:
        packed_e1 = tfc.PackedTensors(f.read())
    tensors_e1 = [string_e1, x_e1_shape, y_e1_shape]
    arrays_e1 = packed_e1.unpack(tensors_e1)
    with open(args.output_folder + '/stream_e2.tfci', "rb") as f:
        packed_e2 = tfc.PackedTensors(f.read())
    tensors_e2 = [string_e2, x_e2_shape, y_e2_shape]
    arrays_e2 = packed_e2.unpack(tensors_e2)
    with open(args.output_folder + '/stream_e3.tfci', "rb") as f:
        packed_e3 = tfc.PackedTensors(f.read())
    tensors_e3 = [string_e3, x_e3_shape, y_e3_shape]
    arrays_e3 = packed_e3.unpack(tensors_e3)

    tensors = [string_B, x_B_shape, y_B_shape, string_e1, x_e1_shape, y_e1_shape, string_e2, x_e2_shape, y_e2_shape, string_e3, x_e3_shape, y_e3_shape]
    arrays = arrays_B + arrays_e1 + arrays_e2 + arrays_e3
    
    
# 实例化网络模块
    with tf.variable_scope('Compress'):
        synthesis_transform_B = SynthesisTransform(args.num_filters_B)
        synthesis_transform_e1 = SynthesisTransform(args.num_filters_e1)
        synthesis_transform_e2 = SynthesisTransform(args.num_filters_e2)
        synthesis_transform_e3 = SynthesisTransform(args.num_filters_e3)
        entropy_bottleneck_B = tfc.EntropyBottleneck(dtype=tf.float32)
        entropy_bottleneck_e1 = tfc.EntropyBottleneck(dtype=tf.float32)
        entropy_bottleneck_e2 = tfc.EntropyBottleneck(dtype=tf.float32)
        entropy_bottleneck_e3 = tfc.EntropyBottleneck(dtype=tf.float32)
        prediction_e1 = PredictionTransform_quality(args.num_filters_B,
                                            args.num_filters_e1, 'a')
        prediction_e2 = PredictionTransform_quality(args.num_filters_e1,
                                            args.num_filters_e2, 'b')
        prediction_e3 = PredictionTransform_quality(args.num_filters_e2,
                                            args.num_filters_e3, 'c')


        y_B_shape = tf.concat([y_B_shape, [args.num_filters_B]], axis=0)
        y_B_hat = entropy_bottleneck_B.decompress(
            string_B, y_B_shape, channels=args.num_filters_B)

        y_e1_shape = tf.concat([y_e1_shape, [args.num_filters_e1]], axis=0)
        y_e1_res_hat = entropy_bottleneck_e1.decompress(
            string_e1, y_e1_shape, channels=args.num_filters_e1)
        y_e1_pred = prediction_e1(y_B_hat)
        y_e1_hat = y_e1_res_hat + y_e1_pred
        y_e2_pred = prediction_e2(y_e1_hat)

        y_e2_shape = tf.concat([y_e2_shape, [args.num_filters_e2]], axis=0)
        y_e2_res_hat = entropy_bottleneck_e2.decompress(
            string_e2, y_e2_shape, channels=args.num_filters_e2)
        y_e2_hat = y_e2_res_hat + y_e2_pred
        y_e3_pred = prediction_e3(y_e2_hat)

        y_e3_shape = tf.concat([y_e3_shape, [args.num_filters_e3]], axis=0)
        y_e3_res_hat = entropy_bottleneck_e3.decompress(
            string_e3, y_e3_shape, channels=args.num_filters_e3)
        y_e3_hat = y_e3_res_hat + y_e3_pred

        x_B_hat = synthesis_transform_B(y_B_hat)
        x_e1_hat = synthesis_transform_e1(tf.concat([y_B_hat, y_e1_hat], -1))
        x_e2_hat = synthesis_transform_e2(tf.concat([y_B_hat, y_e1_hat, y_e2_hat], -1))
        x_e3_hat = synthesis_transform_e3(tf.concat([y_B_hat, y_e1_hat, y_e2_hat, y_e3_hat], -1))

    x_B_hat = x_B_hat[0, :x_B_shape[0], :x_B_shape[1], :]
    x_e1_hat = x_e1_hat[0, :x_e1_shape[0], :x_e1_shape[1], :]
    x_e2_hat = x_e2_hat[0, :x_e2_shape[0], :x_e2_shape[1], :]
    x_e3_hat = x_e3_hat[0, :x_e3_shape[0], :x_e3_shape[1], :]    
    
    op_B = write_png(args.output_folder + '/rec_B.png', x_B_hat)
    op_e1 = write_png(args.output_folder + '/rec_e1.png', x_e1_hat)
    op_e2 = write_png(args.output_folder + '/rec_e2.png', x_e2_hat)
    op_e3 = write_png(args.output_folder + '/rec_e3.png', x_e3_hat)


    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
        tf.train.Saver().restore(sess, save_path=latest)
        sess.run([op_B, op_e1, op_e2, op_e3], feed_dict=dict(zip(tensors, arrays)))


if __name__ == "__main__":
    parser = argparse_flags.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_image', type=str, default='./kodak/kodim01.png', help="Kodak original image path.")
    parser.add_argument('--output_folder', type=str, default='./output', help="Output path.")
    parser.add_argument('--checkpoint_dir', type=str, default='./models_quality/', help="Checkpoint folder")
    parser.add_argument("--num_filters_B", type=int, default=96, help="Number of base layer.")
    parser.add_argument("--num_filters_e1", type=int, default=120, help="Number of filters of enhancement layer e1.")
    parser.add_argument("--num_filters_e2", type=int, default=144, help="Number of filters of enhancement layer e2.")
    parser.add_argument("--num_filters_e3", type=int, default=196, help="Number of filters of enhancement layer e3.")
    

    
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    evaluate(args)
    decompress(args)



        


