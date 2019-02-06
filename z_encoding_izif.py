import os, sys
sys.path.append(os.getcwd())

import cPickle
import csv
from decimal import Decimal
import functools
import pdb
import re
import scipy
import time
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
OUTPUT_DIM = 64*64*1 # Number of pixels in each iamge
ENCODING_SAVE_ITERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000]
ITERS_ENC = 50000 #100000
ENCODING_PLT_ITERS = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000]
N_GPUS = 1

Z_REG_TYPE = None
#Z_REG_TYPE = 'stoch_clip'
#Z_REG_TYPE = 'tanh_fc'
#Z_REG_TYPE = '3s_tanh_fc'
#Z_REG_TYPE = '05s_tanh_fc'
#Z_REG_TYPE = 'hard_clip'
#Z_REG_TYPE = '3s_hard_clip'
#Z_REG_TYPE = '05s_hard_clip'

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
def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def my_Normalize(name, inputs, is_training):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        #return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)
        return tf.layers.batch_normalization(inputs, axis=1, training=is_training, name=name)

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], 1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, is_training=None, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if is_training is not None:
        output = my_Normalize(name+'.BN1', output, is_training)
    else:
        output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if is_training is not None:
        output = my_Normalize(name+'.BN2', output, is_training)
    else:
        output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


def GoodGenerator(n_samples, rand_sampling='normal', noise=None, dim=DIM, nonlinearity=tf.nn.relu, z_out=False, is_training=False, reuse=None):
    with tf.variable_scope('Generator', reuse=reuse):
        if noise is None:
            if rand_sampling == 'unif':
                noise = tf.random_uniform([n_samples, ZDIM], minval=-1., maxval=1.)
            elif rand_sampling == 'normal':
                noise = tf.random_normal([n_samples, ZDIM])

        output = lib.ops.linear.Linear('Generator.Input', ZDIM, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])

        output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, is_training=is_training, resample='up')
        output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, is_training=is_training, resample='up')
        output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, is_training=is_training, resample='up')
        output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, is_training=is_training, resample='up')

        if is_training is not None:
            output = my_Normalize('Generator.OutputN', output, is_training)
        else:
            output = Normalize('Generator.OutputN', [0,2,3], output)
        output = tf.nn.relu(output)
        output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 1, 3, output)
        output = tf.tanh(output)

    if z_out:
        return tf.reshape(output, [-1, OUTPUT_DIM]), noise
    else:
        return tf.reshape(output, [-1, OUTPUT_DIM])


def GoodDiscriminator(inputs, dim=DIM, is_training=False, reuse=None):
    with tf.variable_scope('Discriminator', reuse=reuse):
        output = tf.reshape(inputs, [-1, 1, 64, 64])
        output = lib.ops.conv2d.Conv2D('Discriminator.Input', 1, dim, 3, output, he_init=False)

        output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, is_training=is_training, resample='down')
        output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, is_training=is_training, resample='down')

        output = tf.reshape(output, [-1, 4*4*8*dim])
        out_features = output
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1]), out_features


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
    zdistrib_log_dir = checkpoint_dir_map.replace('checkpoints','zdistrib_log')
    if not os.path.isdir( zdistrib_log_dir ):
        os.makedirs( zdistrib_log_dir )


    # Dataset iterator
    train_gen,test_gen,_ = lib.img_loader.load(BATCH_SIZE)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    (fixed_test_img,) = test_gen().next()


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
                recon_img = Generator(BATCH_SIZE/len(DEVICES), noise=z, is_training=False)
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

                # #################
                img_test = tf.constant( fixed_test_img )
                img_test_rs = tf.reshape(2*((tf.cast(img_test, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])
                enc_z_distrib_test = Encoder(img_test_rs, is_training=False, reuse=True, z_reg_type=Z_REG_TYPE, rand_sampling=rand_sampling)
                # #################

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
        test_writer  = tf.summary.FileWriter(log_dir + '/test')
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


        z_distribution_img = np.zeros((10,0), dtype=np.float32)

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

            if (((iteration+1) % 50 == 0) and ((iteration+1)<=200)) or ((iteration+1) % 500 == 0):
                _enc_z_distrib_test = session.run( enc_z_distrib_test )
                z_hist = np.histogram(_enc_z_distrib_test.flatten(), bins=np.array(range(-5,6)))[0].astype(np.float32).reshape(-1,1)
                z_hist /= z_hist.max()
                z_distribution_img = np.concatenate((z_distribution_img, z_hist), axis=1)
                scipy.misc.imsave('%s/z_distribution_train_enc_izi.png' %(zdistrib_log_dir), z_distribution_img)

            ## -- VALIDATION --
            if ((iteration+1) == 100) or ( (iteration+1) % 500 == 0) or ((iteration+1) in ENCODING_SAVE_ITERS):
                test_map_losses = []
                for n,_test_data in enumerate(test_gen()):
                    (images,) = _test_data
                    _mapping_loss_test, _ml_test_sum = session.run([mapping_loss, merged ], feed_dict={all_real_data_conv: images})
                    test_map_losses.append(_mapping_loss_test)
                    if (n+1)>=nr_valid_btchs:
                        break

                ## -- validation LOGGING **
                lib.plot.plot('z-mapping loss test', np.mean(test_map_losses))
                test_writer.add_summary(_ml_test_sum, iteration+1)
                
                _real_img, _recon_img = session.run([ real_data, recon_img ], feed_dict={all_real_data_conv: fixed_test_img})
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



def anomaly_scoring(checkpoint_dir, checkpoint_iter, dual_iloss=True, nr_mapping_files=9600):
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
    cwd = os.getcwd()
    mapping_path = os.path.join('mappings', model_type_name)

    if not os.path.isdir( mapping_path ):
        os.makedirs( mapping_path )
    
    log_meta_path = os.path.join(mapping_path, 'mapping_results-enc_ckpt_it%d.csv' %checkpoint_iter)


    ## ---- DATA ---------
    _,test_gen,ano_gen = lib.img_loader.load(BATCH_SIZE)
    nr_mapping_batches = nr_mapping_files // BATCH_SIZE


    Generator = GoodGenerator
    Discriminator = GoodDiscriminator

    with tf.Session(config=tf.ConfigProto(device_count={'GPU':len(DEVICES)}, allow_soft_placement=True)) as session:
        real_data = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1, 64, 64])
        real_data_norm = tf.reshape(2*((tf.cast(real_data, tf.float32)/255.)-.5), [BATCH_SIZE, OUTPUT_DIM])

        emb_query     = Encoder(real_data_norm, is_training=False, z_reg_type=z_reg_type, rand_sampling=rand_sampling)  # z
        recon_img = Generator(BATCH_SIZE, noise=emb_query, is_training=False) # img
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

        nr = 1
        encodings = {'target': [], 'zs': []}
        img_dists,z_dists = [],[]

        for is_anom,_gen in enumerate([test_gen(), ano_gen()]): 
            for _idx in xrange(nr_mapping_batches):
                (_data,) = _gen.next()
                _img_dist, _dist_z, _z = session.run([img_distance, z_distance, emb_query ],
                                                  feed_dict={ real_data: _data })

                if np.mod(_idx+1,100)==0:
                    print "%d (of %d) imgs processed ..\timg_m=%.4f\tz_m=%.4f" %( (_idx+1)*BATCH_SIZE,
                                                                        nr_mapping_files,
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
    train_enc_izi("path-to-'checkpoints'-folder-of-WGAN-training", <checkpoint-iter>, loss_type='MSE', kappa=1.0)
    #anomaly_scoring("path-to-'checkpoints'-folder-of-encoder-training", <checkpoint-iter>)



