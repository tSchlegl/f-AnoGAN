"""
Anomaly scoring

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

import cPickle
import csv
import pdb
import re
import time

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



class bcolors:
    PINK = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'



ZDIM = 128
DIM = 64 # Model dimensionality
BATCH_SIZE = 64
OUTPUT_DIM = 64*64*1 # Number of pixels in each image
N_GPUS = 1


print bcolors.GREEN + "\n=== ANOMALY SCORING PARAMETERS ===" + bcolors.ENDC
lib.print_model_settings(locals().copy())

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]


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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
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



def anomaly_scoring(checkpoint_dir, checkpoint_iter, dual_iloss=True):
    if '_unif' in checkpoint_dir:
        rand_sampling = 'unif'
    elif '_norm' in checkpoint_dir:
        rand_sampling = 'normal'

    if 'l2Mean' in checkpoint_dir:
        loss_type = 'l2Mean'
    elif 'l2Sum' in checkpoint_dir:
        loss_type = 'l2Sum'
    elif 'MSE' in checkpoint_dir:
        loss_type = 'MSE'
        kappa = float(re.match(r'.*MSE-k(\d\.\d{2,2})', checkpoint_dir).group(1))
    assert loss_type in checkpoint_dir

    suff_txt = ''
    if dual_iloss: # dual-image-loss .. using both componentes of image-loss for anomaly scoring: residual of images and residual of D-features
        suff_txt = '_dil'

    if '3s_tanh_fc' in checkpoint_dir:
        z_reg_type = '3s_tanh_fc'
    elif '05s_tanh_fc' in checkpoint_dir:
        z_reg_type = '05s_tanh_fc'
    elif 'tanh_fc' in checkpoint_dir:
        z_reg_type = 'tanh_fc'
    elif '3s_hard_clip' in checkpoint_dir:
        z_reg_type = '3s_hard_clip'
    elif '05s_hard_clip' in checkpoint_dir:
        z_reg_type = '05s_hard_clip'
    elif 'hard_clip' in checkpoint_dir:
        z_reg_type = 'hard_clip'
    elif 'stoch_clip' in checkpoint_dir:
        z_reg_type = 'stoch_clip'
    else:
        z_reg_type = None
    print bcolors.YELLOW + "\nUSING z_reg_type='%s'!\n" %z_reg_type + bcolors.ENDC

    
    print bcolors.GREEN + "\nmapping_via_encoder:: checkpoint_dir: %s\ncheckpoint_iter: %d\n"%(checkpoint_dir, checkpoint_iter) + bcolors.ENDC
    model_type_name = checkpoint_dir.replace('z_encoding_d/','').replace('/checkpoints','') + suff_txt
    mapping_path = os.path.join('mappings', model_type_name)

    if not os.path.isdir( mapping_path ):
        os.makedirs( mapping_path )
    
    log_meta_path = os.path.join(mapping_path, 'mapping_results-enc_ckpt_it%d.csv' %checkpoint_iter)


    ## ---- DATA ---------
    test_gen,ano_gen = lib.img_loader.load(BATCH_SIZE, 'anomaly_score')
    nr_mapping_imgs = lib.img_loader.get_nr_test_samples(BATCH_SIZE)


    Generator = GoodGenerator
    Discriminator = GoodDiscriminator

    with tf.Session(config=tf.ConfigProto(device_count={'GPU':len(DEVICES)}, allow_soft_placement=True)) as session:
        real_data = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1, 64, 64])
        real_data_norm = tf.reshape(2*((tf.cast(real_data, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])

        emb_query     = Encoder(real_data_norm, is_training=False, z_reg_type=z_reg_type, rand_sampling=rand_sampling)  # z
        recon_img = Generator(BATCH_SIZE, noise=emb_query, rand_sampling='normal', is_training=False) # img
        _,recon_features = Discriminator(recon_img, is_training=False)
        _,image_features = Discriminator(real_data_norm, is_training=False, reuse=True)

        ##- DISTANCE BASED ON z OF E(Q) AND z OF E(G(E(Q)))
        z_img_emb_query = Encoder( recon_img, is_training=False, reuse=True, z_reg_type=z_reg_type, rand_sampling=rand_sampling ) # z

        ## 
        if (loss_type=='l2Mean') or (loss_type=='l2Sum'):
            img_distance = l2_norm( real_data_norm, recon_img, axis=1 )   # distance based on images
            z_distance = l2_norm( emb_query, z_img_emb_query, axis=1 )   # distance based on z's
        elif loss_type =='MSE':
            loss_img = MSE( real_data_norm, recon_img, axis=1 )
            if dual_iloss:
                loss_fts = MSE( recon_features, image_features, axis=1 )
                img_distance = loss_img + kappa*loss_fts
            else:
                img_distance = loss_img
            z_distance = MSE( emb_query, z_img_emb_query, axis=1 )
        
        
        saver = tf.train.Saver(max_to_keep=15)
        session.run(tf.global_variables_initializer())
        isLoaded, ckpt = load(session, saver, checkpoint_dir, checkpoint_iter)
        assert isLoaded

        start_time = time.time()
        if os.path.isfile(log_meta_path):
            os.remove(log_meta_path)

        encodings = {'target': [], 'zs': []}
        img_dists,z_dists = [],[]

        for is_anom,_gen in enumerate([test_gen(), ano_gen()]):
            nr_mapping_batches = nr_mapping_imgs[is_anom] // BATCH_SIZE
            for _idx in xrange(nr_mapping_batches):
                (_data,) = _gen.next()
                _img_dist, _dist_z, _z = session.run([img_distance, z_distance, emb_query ],
                                                  feed_dict={ real_data: _data })

                if np.mod(_idx+1,100)==0:
                    print "%d (of %d) imgs processed ..\timg_m=%.4f\tz_m=%.4f" %( (_idx+1)*BATCH_SIZE,
                                                                        nr_mapping_batches*BATCH_SIZE,
                                                                        _img_dist.mean(),
                                                                        _dist_z.mean() )

                with open( log_meta_path, "a" ) as f:
                    writer = csv.writer(f, delimiter=',')
                    for di,dz in zip(_img_dist, _dist_z):
                        writer.writerow( [is_anom, di, dz] )

                encodings['target'].append( is_anom )
                encodings['zs'].append( _z )
                img_dists.append( _img_dist )
                z_dists.append( _dist_z )

        took_tm = time.time() - start_time
        print "\nDONE!\t(mapping took %.1f seconds.)\n" %took_tm

        ## --- SAVE RESULTS ---
        with open( log_meta_path.replace('.csv', '.pkl'), 'w') as f:
           cPickle.dump({
                        'encodings': encodings,
                        'img_dists': img_dists,
                        'z_dists': z_dists,
            }, f, cPickle.HIGHEST_PROTOCOL)
        print "Done!\n"



if __name__ == '__main__':
    anomaly_scoring("path-to-'checkpoints'-folder-of-encoder-training", <checkpoint-iter>)
    