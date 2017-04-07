import sys
sys.path.append('..')

import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch

from scipy.misc import imread
from glob import glob
from load import * 

def transform(X):
    return floatX(X)/127.5 - 1

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
    return X

k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 100      # # of examples in batch
npx = 32          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer
ndfc = 1024
nx = npx*npx*nc   # # of dimensions in X
niter = 100       # # of iter at starting learning rate
niter_decay = 100   # # of iter to linearly decay learning rate to zero
ny = 10
margin = 1.

trX, trY, vaX, vaY, teX, teY = load_cifar10()
ntrain, nvalid, ntest = trX.shape[0], vaX.shape[0], teX.shape[0]

desc = 'steingan_supervised'
model_dir = 'models/%s' % desc
samples_dir = 'samples/%s' % desc

dir_list = [model_dir, samples_dir]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)
print desc

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)


gw  = gifn((nz+ny, ngf*4*4*4), 'gw')
gg = gain_ifn((ngf*4*4*4), 'gg')
gb = bias_ifn((ngf*4*4*4), 'gb')
gw2 = gifn((ngf*4+ny, ngf*4, 3, 3), 'gw2')
gg2 = gain_ifn((ngf*4), 'gg2')
gb2 = bias_ifn((ngf*4), 'gb2')
gw3 = gifn((ngf*4+ny, ngf*2, 3, 3), 'gw3')
gg3 = gain_ifn((ngf*2), 'gg3')
gb3 = bias_ifn((ngf*2), 'gb3')
gw4 = gifn((ngf*2+ny, ngf*2, 3, 3), 'gw4')
gg4 = gain_ifn((ngf*2), 'gg4')
gb4 = bias_ifn((ngf*2), 'gb4')
gw5 = gifn((ngf*2+ny, ngf, 3, 3), 'gw5')
gg5 = gain_ifn((ngf), 'gg5')
gb5 = bias_ifn((ngf), 'gb5')
gw6 = gifn((ngf+ny, ngf, 3, 3), 'gw6')
gg6 = gain_ifn((ngf), 'gg6')
gb6 = bias_ifn((ngf), 'gb6')
gwx  = gifn((ngf+ny, nc, 3, 3), 'gwx')

aew1 = difn((ndf, nc, 3, 3), 'aew1') 
aew2 = difn((ndf, ndf, 2, 2), 'aew2') 
aew3 = difn((ndf*2, ndf, 3, 3), 'aew3')
aew4 = difn((ndf*2, ndf*2, 4, 4), 'aew4')
aew5 = difn((ndf*4, ndf*2, 3, 3), 'aew5')
aew6 = difn((ndf*4, ndf*4, 4, 4), 'aew6')

aeg2 = gain_ifn((ndf), 'aeg2') 
aeb2 = bias_ifn((ndf), 'aeb2')
aeg3 = gain_ifn((ndf*2), 'aeg3') 
aeb3 = bias_ifn((ndf*2), 'aeb3')
aeg4 = gain_ifn((ndf*2), 'aeg4') 
aeb4 = bias_ifn((ndf*2), 'aeb4')
aeg5 = gain_ifn((ndf*4), 'aeg5') 
aeb5 = bias_ifn((ndf*4), 'aeb5')
aeg6 = gain_ifn((ndf*4), 'aeg6') 
aeb6 = bias_ifn((ndf*4), 'aeb6')

aeg6t = gain_ifn((ndf*4), 'aeg6t') 
aeb6t = bias_ifn((ndf*4), 'aeb6t')
aeg5t = gain_ifn((ndf*2), 'aeg5t') 
aeb5t = bias_ifn((ndf*2), 'aeb5t')
aeg4t = gain_ifn((ndf*2), 'aeg4t') 
aeb4t = bias_ifn((ndf*2), 'aeb4t')
aeg3t = gain_ifn((ndf), 'aeg3t')
aeb3t = bias_ifn((ndf), 'aeb3t')
aeg2t = gain_ifn((ndf), 'aeg2t')
aeb2t = bias_ifn((ndf), 'aeb2t')

logistic_w = difn((ndf*4,ny), 'logistic_w')
logistic_b = bias_ifn((ny,), 'logistic_b')

gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gw5, gg5, gb5, gw6, gg6, gb6, gwx]
discrim_params = [aew1, aew2, aew3, aew4, aew5, aew6, aeg2, aeb2, aeg3, aeb3, aeg4, aeb4, aeg5, aeb5, aeg6, aeb6, aeg2t, aeb2t, aeg3t, aeb3t, aeg4t, aeb4t, aeg5t, aeb5t, aeg6t, aeb6t, logistic_w, logistic_b]


def gen(Z, Y):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, gw), g=gg, b=gb))
    h = h.reshape((h.shape[0], ngf*4, 4, 4))
    h = conv_cond_concat(h, yb)
    h2 = relu(batchnorm(deconv(h, gw2, subsample=(2, 2), border_mode=(1, 1)), g=gg2, b=gb2))
    h2 = conv_cond_concat(h2, yb)
    h3 = relu(batchnorm(deconv(h2, gw3, subsample=(1, 1), border_mode=(1, 1)), g=gg3, b=gb3))
    h3 = conv_cond_concat(h3, yb)
    h4 = relu(batchnorm(deconv(h3, gw4, subsample=(2, 2), border_mode=(1, 1)), g=gg4, b=gb4))
    h4 = conv_cond_concat(h4, yb)
    h5 = relu(batchnorm(deconv(h4, gw5, subsample=(1, 1), border_mode=(1, 1)), g=gg5, b=gb5))
    h5 = conv_cond_concat(h5, yb)
    h6 = relu(batchnorm(deconv(h5, gw6, subsample=(2, 2), border_mode=(1, 1)), g=gg6, b=gb6))
    h6 = conv_cond_concat(h6, yb)
    x = tanh(deconv(h6, gwx, subsample=(1, 1), border_mode=(1, 1)))

    return x


def discrim(X, Y):

    def classifier(H, Y):
        p_y_given_x = T.nnet.softmax(T.dot(H, logistic_w) + logistic_b)
        neg_lik = -T.sum(T.mul(T.log(p_y_given_x), Y), axis=1)
        return neg_lik, p_y_given_x

    current_input = dropout(X, 0.2) 
    ### encoder ###
    cv1 = relu(dnn_conv(current_input, aew1, subsample=(1,1), border_mode=(1,1)))
    cv2 = relu(batchnorm(dnn_conv(cv1, aew2, subsample=(2,2), border_mode=(0,0)), g=aeg2, b=aeb2))
    cv3 = relu(batchnorm(dnn_conv(cv2, aew3, subsample=(1,1), border_mode=(1,1)), g=aeg3, b=aeb3))
    cv4 = relu(batchnorm(dnn_conv(cv3, aew4, subsample=(4,4), border_mode=(0,0)), g=aeg4, b=aeb4))
    cv5 = relu(batchnorm(dnn_conv(cv4, aew5, subsample=(1,1), border_mode=(1,1)), g=aeg5, b=aeb5))
    cv6 = relu(batchnorm(dnn_conv(cv5, aew6, subsample=(4,4), border_mode=(0,0)), g=aeg6, b=aeb6))

    ### decoder ###
    dv6 = relu(batchnorm(deconv(cv6, aew6, subsample=(4,4), border_mode=(0,0)), g=aeg6t, b=aeb6t)) 
    dv5 = relu(batchnorm(deconv(dv6, aew5, subsample=(1,1), border_mode=(1,1)), g=aeg5t, b=aeb5t))
    dv4 = relu(batchnorm(deconv(dv5, aew4, subsample=(4,4), border_mode=(0,0)), g=aeg4t, b=aeb4t)) 
    dv3 = relu(batchnorm(deconv(dv4, aew3, subsample=(1,1), border_mode=(1,1)), g=aeg3t, b=aeb3t))
    dv2 = relu(batchnorm(deconv(dv3, aew2, subsample=(2,2), border_mode=(0,0)), g=aeg2t, b=aeb2t))
    dv1 = tanh(deconv(dv2, aew1, subsample=(1,1), border_mode=(1,1)))

    hidden = T.flatten(cv6, 2)
    rX = dv1
    mse = T.sqrt(T.sum(T.flatten((X-rX)**2, 2), axis=1))

    #mse = T.sqrt(T.sum(T.abs_(T.flatten(X-rX, 2)),axis=1)) + T.sqrt(T.sum(T.flatten((X-rX)**2, 2), axis=1))
    neg_lik, p_y_given_x = classifier(hidden, Y)
    return hidden, p_y_given_x, rX, mse, neg_lik


def combine_err(mse, neg_lik):

    return mse + T.maximum(margin, neg_lik)


def rbf_kernel(X0):
    xy = T.dot(X0, X0.T)
    x2 = T.sum(X0**2, axis=1).dimshuffle(0, 'x')
    x2e = T.tile(x2, (1, x2.shape[0]))
    H = -2. * xy
    H += x2e
    H += x2e.T
    
    V = H.flatten()
    
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])
    
    h = T.sqrt(0.5 * h / T.log(X0.shape[0].astype('float32') + 1.0)) / 2.
    Kxy = T.exp(-H / h ** 2 / 2.0)
    
    neighbors = T.argsort(H, axis=1)[:, 1]
    return Kxy, neighbors, h


def gradient(X0, Y):

    _, _, _, mse, neg_lik = discrim(X0, Y)
    grad = -1.0 * T.grad(combine_err(mse, neg_lik).mean() , X0)

    return grad


def svgd_gradient(X0, Y):

    hidden, _, _, mse, neg_lik = discrim(X0, Y)
    grad = -1.0 * T.grad( combine_err(mse, neg_lik).mean(), X0)

    kxy, neighbors, h = rbf_kernel(hidden)  #TODO

    coff = T.exp( - T.sum((hidden[neighbors] - hidden)**2, axis=1) / h**2 / 2.0 )
    v = coff.dimshuffle(0, 'x') * (-hidden[neighbors] + hidden) / h**2

    X1 = X0[neighbors]
    hidden1, _, _, _, _ = discrim(X1, Y)
    dxkxy = T.Lop(hidden1, X1, v)

    #svgd_grad = (T.dot(kxy, T.flatten(grad, 2)).reshape(dxkxy.shape) + dxkxy) / T.sum(kxy, axis=1).dimshuffle(0, 'x', 'x', 'x')
    svgd_grad = grad + dxkxy / 2.
    return grad, svgd_grad


X = T.tensor4() # data

X0 = T.tensor4() # vgd samples

deltaX = T.tensor4() #vgd gradient 

Z = T.matrix()

Y = T.matrix()

epsilon = T.tensor4()


### define discriminative cost ###
H_data, py_data, rX_data, mse_data, neg_lik_data = discrim(X, Y)
H_vgd, py_data, rX_vgd, mse_vgd, neg_lik_vgd = discrim(X0, Y)

cost_data = combine_err(mse_data, neg_lik_data).mean()
cost_vgd = combine_err(mse_vgd, neg_lik_vgd).mean()

balance_weight = sharedX(.3)
d_cost = cost_data - balance_weight * cost_vgd   # for discriminative model, minimize cost

gX = gen(Z, Y)
g_cost = -1 * T.sum(T.sum(T.flatten(gX, 2) * T.flatten(deltaX, 2), axis=1)) #update generate models by minimize reconstruct mse

cost = [mse_data.mean(), neg_lik_data.mean(), mse_vgd.mean(), neg_lik_vgd.mean()]

d_lr = 5e-4
g_lr = 1e-3

d_lrt = sharedX(d_lr)
g_lrt = sharedX(g_lr)

d_updater = updates.Adam(lr=d_lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=g_lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))

d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)


print 'COMPILING'
t = time()
_gen = theano.function([Z, Y], gX)
_discrim = theano.function([X, Y], discrim(X, Y))
_train_d = theano.function([X, X0, Y], cost, updates=d_updates)
_train_g = theano.function([Z, Y, deltaX], g_cost, updates=g_updates)
_svgd_gradient = theano.function([X0, Y], svgd_gradient(X0, Y))
_gradient = theano.function([X0, Y], gradient(X0, Y))
_reconstruction_cost = theano.function([X], T.mean(mse_data))
_reconstruction = theano.function([X], rX_data)
print '%.2f seconds to compile theano functions'%(time()-t)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(200, nz)))
sample_ymb = floatX(OneHot(np.asarray([[i for _ in range(20)] for i in range(10)]).flatten(), ny))

n_updates = 0
n_epochs = 0

t = time()
for epoch in range(1, 200+1):

    print 'cifar 10, steigan, %s, iter %d' % (desc, epoch)
    trX, trY = shuffle(trX, trY)

    for imb, ymb in tqdm(iter_data(trX, trY, size=nbatch), total=ntrain/nbatch):
        imb = transform(imb.reshape(imb.shape[0], nc, npx, npx))
        ymb = floatX(OneHot(ymb, ny))
        zmb = floatX(np_rng.uniform(-1., 1., size=(imb.shape[0], nz)))
        # generate samples
        samples = _gen(zmb, ymb)

        # G
        grad, vgd_grad = _svgd_gradient(samples, ymb)
        _train_g(zmb, ymb, floatX(vgd_grad)) 

        # D
        cost = _train_d(imb, samples, ymb)

        n_updates += 1

        cost_batch_vgd = _reconstruction_cost(floatX(samples))
        cost_batch_data = _reconstruction_cost(imb)

        if n_updates % 10 == 0:
            print desc, cost_batch_data, cost_batch_vgd

        if cost_batch_data > cost_batch_vgd:
            balance_weight.set_value(0.3)
        else:
            balance_weight.set_value(0.1)


    samples = np.asarray(_gen(sample_zmb, sample_ymb))
    color_grid_vis(inverse_transform(samples), (10, 20), 'samples/%s/gan-%d.png' % (desc, epoch))
    color_grid_vis(inverse_transform(_reconstruction(imb)), (10, 10), 'samples/%s/ae-%d.png' % (desc, epoch))

    n_epochs += 1

    if epoch % 50 == 0:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, epoch))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, epoch))

print '%.2f seconds to train the generative model' % (time()-t)
print 'DONE'

