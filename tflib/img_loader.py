"""
Load images and provide splits (train_normal, test_normal, and test_anom(alous)) as arrays

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



from glob import glob
import numpy as np
import os
import pdb
import scipy.misc
import time


trainset_path     = "path-to-folder-holding-normal-training-images"
trainset_val_path = "path-to-folder-holding-normal-validation-images"
test_normal_path  = "path-to-folder-holding-normal-test-images"
test_anom_path    = "path-to-folder-holding-anom-test-images"


def get_files(data_set):
        if data_set == 'train_normal':
            return glob(os.path.join(trainset_path, "*.png"))
        if data_set == 'valid_normal':
            return glob(os.path.join(trainset_val_path, "*.png"))
        elif data_set == 'test_normal':
            return glob(os.path.join(test_normal_path, "*.png"))
        elif data_set == 'test_anom':
            return glob(os.path.join(test_anom_path, "*.png"))

def get_nr_training_samples(batch_size):
    files = glob(os.path.join(trainset_path, "*.png"))
    total_nr_samples = len(files)
    nr_training_samples = total_nr_samples - np.mod(total_nr_samples, batch_size)

    return nr_training_samples

def get_nr_samples(data_set, batch_size):
    files = get_files(data_set)
    total_nr_samples = len(files)
    nr_samples = total_nr_samples - np.mod(total_nr_samples, batch_size)

    return nr_samples

def get_nr_test_samples(batch_size):
    return ( get_nr_samples('test_normal', batch_size),
             get_nr_samples('test_anom', batch_size) 
            )

def make_generator(data_set, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 1, 64, 64), dtype='int32')

        files = get_files(data_set)
        assert(len(files) > 0)

        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, f in enumerate(files):
            image = scipy.misc.imread(f, mode='L')
            if np.random.rand()>=0.5:
                image = image[:,::-1]
            images[n % batch_size] = np.expand_dims( image, 0)
            
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch


def make_ad_generator(data_set, batch_size):
    def get_epoch():
        images = np.zeros((batch_size, 1, 64, 64), dtype='int32')

        files = get_files(data_set)
        nr_files = len(files)
        assert(nr_files > 0)

        for n, f in enumerate(files):
            image = scipy.misc.imread(f, mode='L')
            images[n % batch_size] = np.expand_dims( image, 0)

            if (n+1) % batch_size == 0:
                yield (images,)
            elif (n+1)==nr_files:
                final_btchsz = (n%batch_size)+1
                yield (images[:final_btchsz],)
    return get_epoch


def load(batch_size, run_type):
    if 'train' in run_type:
        return (
            make_generator('train_normal', batch_size),
            make_generator('valid_normal', batch_size)
        )
    elif run_type=='anomaly_score':
        return (
            make_ad_generator('test_normal', batch_size),
            make_ad_generator('test_anom', batch_size)
        )


if __name__ == '__main__':
    train_gen, valid_gen = load(16, 'encoder_train')
    t0 = time.time()
    for n, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if n == 1000:
            break
        t0 = time.time()