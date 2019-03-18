import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [1]
def tick():
	_iter[0] += 1

def tickit(iteration):
	_iter[0] = iteration

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush(plt_dir):
	prints = []

	for name, vals in _since_last_flush.items():
		prints.append("{}\t{}".format(name, np.mean(vals.values())))
		_since_beginning[name].update(vals)

		x_vals = np.sort(_since_beginning[name].keys())
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		#plt.yscale('log')
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig('%s/%s.jpg' %(plt_dir,name.replace(' ', '_')))

	print "iter {}\t{}".format(_iter[0], "\t".join(prints))
	_since_last_flush.clear()

	with open('%s/log.pkl'%plt_dir, 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)