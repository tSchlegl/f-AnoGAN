"""
Encoder (izi_f) training

Copyright (c) 2018 Thomas Schlegl

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


import os, sys
sys.path.append(os.getcwd())

import pdb
import re
import time
from tqdm import tqdm

from wgangp_64x64 import GoodGenerator, GoodDiscriminator, ResidualBlock

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.img_loader
import tflib.ops.layernorm
import tflib.plot



class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'


MODE = 'wgan-gp'
ZDIM = 128
DIM = 64 # Model dimensionality
BATCH_SIZE = 64
OUTPUT_DIM = 64*64*1 # Number of pixels in each image
ENCODING_SAVE_ITERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
ITERS_ENC = 50000 #100000
N_GPUS = 1

Z_REG_TYPE = 'tanh_fc' # DEFAULT (paper version)
#Z_REG_TYPE = None     # unconstrained ('linear') encoder training (see Appendix A.2.)
#
##Z_REG_TYPE = 'stoch_clip'
##Z_REG_TYPE = '3s_tanh_fc'
##Z_REG_TYPE = '05s_tanh_fc'
##Z_REG_TYPE = 'hard_clip'
##Z_REG_TYPE = '3s_hard_clip'
##Z_REG_TYPE = '05s_hard_clip'

print bcolors.GREEN + "\n=== ENCODER TRAINING PARAMETERS ===" + bcolors.ENDC
lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]

timestamp = time.strftime("%Y-%m-%d-%H%M")

z_reg_type_txt = ''
if Z_REG_TYPE is not None:
    z_reg_type_txt = '_%s' %Z_REG_TYPE

## -- my loss functions
def l2_norm(x, y, axis=None):
    if axis is None:
        return tf.reduce_sum(tf.pow(x-y, 2))
    else:
        return tf.reduce_sum(tf.pow(x-y, 2), axis=axis)


def MSE(x, y, axis=None):
    if axis is None:
        return tf.reduce_mean(tf.pow(x-y, 2))
    else:
        return tf.reduce_mean(tf.pow(x-y, 2), axis=axis)
## --

def save(session, saver, checkpoint_dir, step):
    print(" [*] Saving checkpoint (step %d) ..." %step)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save( session,
                os.path.join(checkpoint_dir, "model"),
                global_step=step)


def load(session, saver, checkpoint_dir, checkpoint_iter=None):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if checkpoint_iter is not None:
            last_ckpt_iter = re.match(r'.*.model-(\d+)', ckpt.model_checkpoint_path).group(1)
            target_ckpt_path = re.sub( 'model-%s'%last_ckpt_iter, 'model-%d'%checkpoint_iter, ckpt.model_checkpoint_path)
            saver.restore(session, target_ckpt_path)
            idxx = target_ckpt_path.rfind('/')
            ckpt_name = target_ckpt_path[idxx+1:]
        else:
            saver.restore(session, ckpt.model_checkpoint_path)
            idxx = ckpt.model_checkpoint_path.rfind('/')
            ckpt_name = ckpt.model_checkpoint_path[idxx+1:]
        return True, ckpt_name
    else:
        return False, ''


def get_ckpt_name(checkpoint_dir, checkpoint_iter=None):
    print(" [*] Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        if checkpoint_iter is not None:
            last_ckpt_iter = re.match(r'.*.model-(\d+)', ckpt.model_checkpoint_path).group(1)
            target_ckpt_path = re.sub( last_ckpt_iter, str(checkpoint_iter), ckpt.model_checkpoint_path)
            idxx = target_ckpt_path.rfind('/')
            ckpt_name = target_ckpt_path[idxx+1:]
        else:
            idxx = ckpt.model_checkpoint_path.rfind('/')
            ckpt_name = ckpt.model_checkpoint_path[idxx+1:]
        return ckpt_name


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def my_Normalize(name, inputs, is_training):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        #return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)
        return tf.layers.batch_normalization(inputs, axis=1, training=is_training, name=name)


def Encoder(inputs, is_training, dim=DIM, z_dim=ZDIM, rand_sampling='normal', reuse=None, z_reg_type=None, denoise=None):
    with tf.variable_scope('Encoder', reuse=reuse):
        if denoise is not None:
            inputs = tf.nn.dropout(inputs, keep_prob=denoise)
        output = tf.reshape(inputs, [-1, 1, 64, 64])
        output = lib.ops.conv2d.Conv2D('Encoder.Input', 1, dim, 3, output, he_init=False)

        output = ResidualBlock('Encoder.Res1', dim, 2*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Encoder.Res2', 2*dim, 4*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Encoder.Res3', 4*dim, 8*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Encoder.Res4', 8*dim, 8*dim, 3, output, is_training=is_training, resample='down')

        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Encoder.Output', 4*4*8*dim, z_dim, output)

    if z_reg_type is None:
        return output
    elif z_reg_type == 'tanh_fc':
        return tf.nn.tanh( output )
    elif z_reg_type == '3s_tanh_fc':
        return tf.nn.tanh( output ) * 3
    elif z_reg_type == '05s_tanh_fc':
        return tf.nn.tanh( output ) * 0.5
    elif z_reg_type == 'hard_clip':
        return tf.clip_by_value( output, -1., 1. )
    elif z_reg_type == '3s_hard_clip':
        return tf.clip_by_value( output, -3., 3. )
    elif z_reg_type == '05s_hard_clip':
        return tf.clip_by_value( output, -0.5, 0.5 )
    elif z_reg_type == 'stoch_clip': ## IMPLEMENTS STOCHASTIC CLIPPING  -->> https://arxiv.org/pdf/1702.04782.pdf
        if rand_sampling == 'unif':
            condition = tf.greater(tf.abs(output), 1.)
            true_case = tf.random_uniform(output.get_shape(), minval=-1., maxval=1.)
        elif rand_sampling == 'normal':
            condition = tf.greater(tf.abs(output), 3.)
            true_case = tf.random_normal(output.get_shape())
            print bcolors.YELLOW + "\nImplementing STOCH-CLIP with NORMAL z-mapping!\n" + bcolors.ENDC
        return tf.where(condition, true_case, output)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


def train_enc_izi(checkpoint_dir, checkpoint_iter, nr_valid_plt=9, nr_valid_btchs=10, lr=5e-5, loss_type='MSE', kappa=1.0, denoise=None):    
    np.random.seed(1234)

    if '_unif' in checkpoint_dir:
        rand_sampling = 'unif'
    elif '_norm' in checkpoint_dir:
        rand_sampling = 'normal'

    loss_type_txt = loss_type
    if loss_type == 'MSE':
        loss_type_txt = '%s-k%.2f' %(loss_type, kappa)

    if lr!=5e-5:
        loss_type_txt = '%s-lr%.0E' %(loss_type_txt, lr)

    if denoise is not None:
        dn_txt = "_dn%.1f"%denoise
    else:
        dn_txt = ""

    scriptname = os.path.basename(__file__).replace('.pyc', '').replace('.py', '')
    checkpoint_dir_map = os.path.join('z_encoding_d/izi%s%s'%(z_reg_type_txt,dn_txt), checkpoint_dir.replace('wganTrain','wganTrain-iter%d'%checkpoint_iter).replace('/checkpoints', '-enc_iter%d_%s'%(ITERS_ENC,timestamp)), "%s-%s%s"%(scriptname,loss_type_txt,z_reg_type_txt), 'checkpoints')
    if not os.path.isdir( checkpoint_dir_map ):
        os.makedirs( checkpoint_dir_map )
    samples_dir_map = checkpoint_dir_map.replace('checkpoints','samples')
    if not os.path.isdir( samples_dir_map ):
        os.makedirs( samples_dir_map )
    log_dir = checkpoint_dir_map.replace('checkpoints','log_dir')
    if not os.path.isdir( log_dir ):
        os.makedirs( log_dir )


    # Dataset iterator
    train_gen,valid_gen = lib.img_loader.load(BATCH_SIZE, 'encoder_train')

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    (fixed_valid_img,) = valid_gen().next()


    Generator = GoodGenerator
    Discriminator = GoodDiscriminator

    with tf.Session(config=tf.ConfigProto(device_count={'GPU':len(DEVICES)}, allow_soft_placement=True)) as session:
        all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1, 64, 64])
        if tf.__version__.startswith('1.'):
            split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
            print "\n\nDEVICES: %s\n\n" %DEVICES
        else:
            split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
        mapping_losses = []
        mapping_losses_img = []
        mapping_losses_fts = []

        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):
                real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE/len(DEVICES), OUTPUT_DIM])

                z = Encoder(real_data, is_training=True, z_reg_type=Z_REG_TYPE, rand_sampling=rand_sampling, denoise=denoise)
                recon_img = Generator(BATCH_SIZE/len(DEVICES), noise=z, rand_sampling='normal', is_training=False)
                _,recon_features = Discriminator(recon_img, is_training=False)
                _,image_features = Discriminator(real_data, is_training=False, reuse=True)

                if loss_type=='l2Mean':
                    loss = tf.reduce_mean(l2_norm( real_data, recon_img, axis=1 ))
                elif loss_type=='l2Sum':
                    raise
                    loss = l2_norm( real_data, recon_img )
                elif loss_type =='MSE':
                    loss_img = MSE( real_data, recon_img )
                    loss_fts = MSE( recon_features, image_features )
                    loss = loss_img + kappa*loss_fts

                mapping_losses.append(loss)
                mapping_losses_img.append(loss_img)
                mapping_losses_fts.append(loss_fts)


        mapping_loss = tf.add_n(mapping_losses) / len(DEVICES)
        mapping_loss_img = tf.add_n(mapping_losses_img) / len(DEVICES)
        mapping_loss_fts = tf.add_n(mapping_losses_fts) / len(DEVICES)


        ## ****************************************************
        ## *** TENSORBOARD
        ## ****************************************************
        mapping_loss_sum = tf.summary.scalar("mapping_loss", mapping_loss)
        mapping_loss_img_sum = tf.summary.scalar("mapping_loss_img", mapping_loss_img)
        mapping_loss_fts_sum = tf.summary.scalar("mapping_loss_fts", mapping_loss_fts)
        merged = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter(log_dir + '/train', session.graph)
        valid_writer  = tf.summary.FileWriter(log_dir + '/valid')
        ## ****************************************************

        t_vars = tf.trainable_variables()
        encoder_vars = [var for var in t_vars if 'Encoder' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(mapping_loss,
                                                 var_list=encoder_vars, colocate_gradients_with_ops=True)

        saver = tf.train.Saver(max_to_keep=15)
        session.run(tf.global_variables_initializer())

        ckpt = get_ckpt_name(checkpoint_dir, checkpoint_iter)
        ckpt_pth = os.path.join(checkpoint_dir, ckpt)
        varmap = {}
        for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if ('Generator' in item.name) or ('Discriminator' in item.name):
                varmap[item.name.replace(':0', '')] = item

        saver_restore = tf.train.Saver(varmap)
        saver_restore.restore(session, ckpt_pth)

        gen = inf_train_gen()
        for iteration in tqdm(xrange(ITERS_ENC)):
            start_time = time.time()
            _data = gen.next()

            ## -- TRAIN encoder --
            _mapping_loss, _, _ml_sum = session.run([mapping_loss, train_op, merged ], feed_dict={all_real_data_conv: _data})
            
            ## -- train LOGGING **
            lib.plot.plot('z-mapping loss train', _mapping_loss)
            lib.plot.plot('time', time.time() - start_time)
            
            if (iteration+1) % 50 == 0:
                train_writer.add_summary(_ml_sum, iteration+1)

            ## -- VALIDATION --
            if ((iteration+1) == 100) or ( (iteration+1) % 500 == 0) or ((iteration+1) in ENCODING_SAVE_ITERS):
                valid_map_losses = []
                for n,_valid_data in enumerate(valid_gen()):
                    (images,) = _valid_data
                    _mapping_loss_valid, _ml_valid_sum = session.run([mapping_loss, merged ], feed_dict={all_real_data_conv: images})
                    valid_map_losses.append(_mapping_loss_valid)
                    if (n+1)>=nr_valid_btchs:
                        break

                ## -- validation LOGGING **
                lib.plot.plot('z-mapping loss valid', np.mean(valid_map_losses))
                valid_writer.add_summary(_ml_valid_sum, iteration+1)
                
                _real_img, _recon_img = session.run([ real_data, recon_img ], feed_dict={all_real_data_conv: fixed_valid_img})
                _real_img = _real_img.reshape((-1, 1, 64, 64))
                inputs_plt = ((_real_img[:nr_valid_plt]+1.)*(255.99/2)).astype('int32')
                _recon_img = _recon_img.reshape((-1, 1, 64, 64))
                recons_plt = ((_recon_img[:nr_valid_plt]+1.)*(255.99/2)).astype('int32')
                img_pairs  = np.concatenate( (inputs_plt,recons_plt), axis=2 )
                lib.save_images.save_images_as_row(img_pairs, '{}/samples_{}.png'.format(samples_dir_map, iteration+1))

            if (iteration < 3) or ( (iteration+1) % 500 == 0) or ((iteration+1) in ENCODING_SAVE_ITERS):
                lib.plot.flush(log_dir)

            lib.plot.tick()

            if (iteration+1) in ENCODING_SAVE_ITERS:
                save(session, saver, checkpoint_dir_map, iteration+1)

        # SAVE FINAL MODEL
        save(session, saver, checkpoint_dir_map, iteration+1)



if __name__ == '__main__':
    train_enc_izi("path-to-'checkpoints'-folder-of-WGAN-training", <checkpoint-iter>, loss_type='MSE', kappa=1.0)
