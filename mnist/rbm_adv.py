import sys
sys.path.append('..')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import random
import os
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib

import theano
import theano.tensor as T

from lib import activations
from lib import inits
from lib import updates
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng, t_rng
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout

from logz.rbm_hais_logz import  ais_logZ
from load import mnist_with_valid_set


desc = 'rbm_adv' 
model_dir = 'models/%s' % desc
samples_dir = 'samples/%s' % desc

dir_list = [model_dir, samples_dir]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)


''' load mnist data'''
trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()
trX, vaX, teX = trX/255., vaX/255., teX/255.
ntrain, nvalid, ntest = len(trX), len(vaX), len(teX)
print ntrain, nvalid, ntest

def transform(X):
    return (floatX(X)).reshape(-1, nc, npx, npx)

def inverse_transform(X):
    X = X.reshape(-1, npx, npx)
    return X


lr = 3e-4  # learning rate for model 
b1 = .5
l2 = 1e-5
nc = 1            # # of channels in image
ny = 10           # # of classes
nbatch = 100      # # of examples in batch
npx = 28          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngfc = 1024       # # of gen units for fully connected layers
ndfc = 1024       # # of discrim units for fully connected layers
ngf = 64          # # of gen filters in first conv layer
ndf = 64          # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100   # # of iter to linearly decay learning rate to zero


n_hidden = 10
n_observe = trX.shape[1]

r_gifn = inits.Uniform(scale=4*np.sqrt(6./(n_observe+n_hidden)))
r_bias_fn = inits.Constant()

gB = r_gifn((n_observe, n_hidden), 'gB') 
gb = r_bias_fn((n_observe,), 'gb') 
gc = r_bias_fn((n_hidden,), 'gc') 
rbm_params = [gB, gb, gc]


relu = activations.Rectify()
sigmoid = activations.Sigmoid()
tanh = activations.Tanh()
lrelu = activations.LeakyRectify()
bce = T.nnet.binary_crossentropy

gifn = inits.Normal(scale=0.02)

gw  = gifn((nz, ngfc), 'gw')
gw2 = gifn((ngfc, ngf*2*7*7), 'gw2')
gw3 = gifn((ngf*2, ngf, 5, 5), 'gw3')
gwx = gifn((ngf, nc, 5, 5), 'gwx')

gen_params = [gw, gw2, gw3, gwx]


def gen(Z, w, w2, w3, gwx):
    h = relu(batchnorm(T.dot(Z, w)))
    h2 = relu(batchnorm(T.dot(h, w2)))
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))

    x = sigmoid(deconv(h3, gwx, subsample=(2, 2), border_mode=(2, 2)))
    return x


def rbf_kernel(X):

    XY = T.dot(X, X.T)
    x2 = T.sum(X**2, axis=1).dimshuffle(0, 'x')
    X2e = T.repeat(x2, X.shape[0], axis=1)
    H = X2e +  X2e.T - 2. * XY

    V = H.flatten()
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])

    h = T.sqrt(.5 * h / T.log(H.shape[0].astype('float32') + 1.)) 
    
    # compute the rbf kernel
    kxy = T.exp(-H / (h ** 2) / 2.0)

    dxkxy = -T.dot(kxy, X)
    sumkxy = T.sum(kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = T.add(dxkxy, T.mul(X, sumkxy)) / (h ** 2)

    return kxy, dxkxy


def discrim(X):
    return logp_rbm(X)

def logp_rbm(X):
    y = T.dot(X, gB) + gc  
    y_max = T.max(T.maximum(y, -y), axis=1).dimshuffle(0,'x')
    log_sum = y_max + T.log(T.exp(y - y_max) + T.exp(-y - y_max)) # apply the log sum trick
    log_sum = T.sum(log_sum, axis=1)
    logp = T.dot(X, gb.dimshuffle(0, 'x')).flatten() - .5 * T.sum(X*X, axis=1) + log_sum

    return logp


def dlogp_rbm(X):
    y = T.dot(X, gB) + gc
    phi = 1. - 2. / (1+T.exp(2*y))
    score = gb - X + T.dot(phi, gB.T)
    return score


def svgd_gradient(X):
    grad = dlogp_rbm(X)
    kxy, dxkxy = rbf_kernel(X)
    svgd_grad = (T.dot(kxy, grad) + dxkxy) / T.sum(kxy, axis=1).dimshuffle(0, 'x') 

    return grad, svgd_grad


X = T.matrix()

X0 = T.matrix() # samples

# vgd gradient 
deltaX = T.tensor4()

# random noise 
Z = T.matrix()

f_real = discrim(X)   # data
f_gen = discrim(X0)   # vgd particles 

cost_data = -1 * f_real.mean()
cost_vgd = -1 * f_gen.mean()

gX = gen(Z, *gen_params)
g_cost = -1 * T.sum(T.sum(T.flatten(gX, 2) * T.flatten(deltaX, 2), axis=1)) #update generate models by minimize reconstruct mse

balance_weight = sharedX(1.)
d_cost = cost_data - balance_weight * cost_vgd   # for discriminative model, minimize cost

lrt = sharedX(lr)
d_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))

d_updates = d_updater(rbm_params, d_cost)
g_updates = g_updater(gen_params, g_cost)

print 'COMPILING'
t = time()
_train_d = theano.function([X, X0], d_cost, updates=d_updates)
_train_g = theano.function([Z, deltaX], g_cost, updates=g_updates)
_gen = theano.function([Z], gen(Z, *gen_params))
_logp_rbm = theano.function([X], logp_rbm(X))
_svgd_gradient = theano.function([X], svgd_gradient(X))
print '%.2f seconds to compile theano functions'%(time()-t)

nbatch = 100
n_iter = 20
n_updates = 0

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(200, nz)))

for iter in tqdm(range(1, n_iter+1)):
    trX = shuffle(trX)
    for imb in iter_data(trX, size=nbatch):
        imb = floatX(imb)
        zmb = floatX(np_rng.uniform(-1., 1., size=(nbatch, nz)))

        # generate samples
        samples = floatX(_gen(zmb).reshape(-1, nx))

        grad, svgd_grad = _svgd_gradient(samples)
        _train_g(zmb, floatX(svgd_grad.reshape(-1, nc, npx, npx))) # generator

        _train_d(imb, floatX(samples))  # discriminator

        n_updates += 1

    if iter % 50 == 0:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, iter))
        joblib.dump([p.get_value() for p in rbm_params], 'models/%s/%d_rbm_params.jl'%(desc, iter))

    samples = np.asarray(_gen(sample_zmb))
    grayscale_grid_vis(inverse_transform(samples), (10, 20), '%s/%d.png' % (samples_dir, iter))

# adversarial
logz_approx = ais_logZ(gB.get_value(), gb.get_value(), gc.get_value())
ll_train = _logp_rbm(floatX(trX)) - logz_approx
ll_test = _logp_rbm(floatX(teX)) - logz_approx
print iter, 'train', np.mean(ll_train), 'test', ll_test.mean(), 'logz', logz_approx

#np.savez('adv_ll_train.npz', ll=ll_train)
#np.savez('adv_ll_test.npz', ll=ll_test)

print 'DONE!'


