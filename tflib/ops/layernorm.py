"""
Copyright (c) 2017 Ishaan Gulrajani

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

def Layernorm(name, norm_axes, inputs):
    mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

    # Assume the 'neurons' axis is the first of norm_axes. This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[norm_axes[0]]

    offset = lib.param(name+'.offset', np.zeros(n_neurons, dtype='float32'))
    scale = lib.param(name+'.scale', np.ones(n_neurons, dtype='float32'))

    # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    offset = tf.reshape(offset, [-1] + [1 for i in xrange(len(norm_axes)-1)])
    scale = tf.reshape(scale, [-1] + [1 for i in xrange(len(norm_axes)-1)])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result