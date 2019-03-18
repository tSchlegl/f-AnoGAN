"""
2D convolution

Copyright (c) 2017 Ishaan Gulrajani
Copyright (c) 2018 Thomas Schlegl ... minor modifications

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tflib as lib

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib


_default_weightnorm = False
def enable_default_weightnorm():
    global _default_weightnorm
    _default_weightnorm = True

_weights_stdev = None
def set_weights_stdev(weights_stdev):
    global _weights_stdev
    _weights_stdev = weights_stdev

def unset_weights_stdev():
    global _weights_stdev
    _weights_stdev = None

def Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=True, mask_type=None, stride=1, weightnorm=None, biases=True, gain=1.):
    """
    inputs: tensor of shape (batch size, num channels, height, width)
    mask_type: one of None, 'a', 'b'

    returns: tensor of shape (batch size, num channels, height, width)
    """
    with tf.name_scope(name) as scope:

        if mask_type is not None:
            mask_type, mask_n_channels = mask_type

            mask = np.ones(
                (filter_size, filter_size, input_dim, output_dim), 
                dtype='float32'
            )
            center = filter_size // 2

            # Mask out future locations
            # filter shape is (height, width, input channels, output channels)
            mask[center+1:, :, :, :] = 0.
            mask[center, center+1:, :, :] = 0.

            # Mask out future channels
            for i in xrange(mask_n_channels):
                for j in xrange(mask_n_channels):
                    if (mask_type=='a' and i >= j) or (mask_type=='b' and i > j):
                        mask[
                            center,
                            center,
                            i::mask_n_channels,
                            j::mask_n_channels
                        ] = 0.


        def uniform(stdev, size):
            return np.random.uniform(
                low=-stdev * np.sqrt(3),
                high=stdev * np.sqrt(3),
                size=size
            ).astype('float32')

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if mask_type is not None: # only approximately correct
            fan_in /= 2.
            fan_out /= 2.

        if he_init:
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))

        if _weights_stdev is not None:
            filter_values = uniform(
                _weights_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )
        else:
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim)
            )

        # print "WARNING IGNORING GAIN"
        filter_values *= gain

        filters = lib.param(name+'.Filters', filter_values)

        if weightnorm==None:
            weightnorm = _default_weightnorm
        if weightnorm:
            norm_values = np.sqrt(np.sum(np.square(filter_values), axis=(0,1,2)))
            target_norms = lib.param(
                name + '.g',
                norm_values
            )
            with tf.name_scope('weightnorm') as scope:
                norms = tf.sqrt(tf.reduce_sum(tf.square(filters), reduction_indices=[0,1,2]))
                filters = filters * (target_norms / norms)

        if mask_type is not None:
            with tf.name_scope('filter_mask'):
                filters = filters * mask

        '''
        if device_lib.list_local_devices()[0].device_type=='CPU':
            dataf = 'NHWC'
        else:
            dataf = 'NCHW'
        '''

        result = tf.nn.conv2d(
            input=inputs, 
            filter=filters, 
            strides=[1, 1, stride, stride],
            padding='SAME',
            #data_format=dataf
            data_format='NCHW'
        )

        if biases:
            _biases = lib.param(
                name+'.Biases',
                np.zeros(output_dim, dtype='float32')
            )

            result = tf.nn.bias_add(result, _biases, data_format='NCHW')


        return result
