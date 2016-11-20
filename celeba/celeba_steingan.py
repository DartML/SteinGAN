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
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool

from lib import activations
from lib import updates
from lib import inits
from lib.vis import color_grid_vis
from lib.rng import py_rng, np_rng
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout, l2normalize
from lib.metrics import nnc_score, nnd_score
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data, center_crop, patch
from lasagne.layers import InverseLayer

from scipy.misc import imread
from glob import glob

def transform(X):
    #X = [center_crop(x, npx) for x in X]
    return floatX(X).transpose(0, 3, 1, 2)/127.5 - 1.

def inverse_transform(X):
    X = (X.reshape(-1, nc, npx, npx).transpose(0, 2, 3, 1)+1.)/2.
    return X

k = 1             # # of discrim updates for each gen update
l2 = 1e-5         # l2 weight decay
nvis = 196        # # of samples to visualize during training
b1 = 0.5          # momentum term of adam
nc = 3            # # of channels in image
nbatch = 100      # # of examples in batch
npx = 64          # # of pixels width/height of images
nz = 100          # # of dim for Z
ngf = 128         # # of gen filters in first conv layer
ndf = 128         # # of discrim filters in first conv layer
nx = npx*npx*nc   # # of dimensions in X
niter = 20        # # of iter at starting learning rate
niter_decay = 0   # # of iter to linearly decay learning rate to zero


t = time()
npzfiles = glob(os.path.join('dataset', 'faces', '*.npz'))


desc = 'celeba_steingan'
model_dir = 'models/%s' % desc
samples_dir = 'samples/%s' % desc

dir_list = [model_dir, samples_dir]
for dir in dir_list:
    if not os.path.exists(dir):
        os.makedirs(dir)

relu = activations.Rectify()
sigmoid = activations.Sigmoid()
lrelu = activations.LeakyRectify()
tanh = activations.Tanh()
bce = T.nnet.binary_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)


gw  = gifn((nz, ngf*8*4*4), 'gw')
gg = gain_ifn((ngf*8*4*4), 'gg')
gb = bias_ifn((ngf*8*4*4), 'gb')
gw2 = gifn((ngf*8, ngf*4, 5, 5), 'gw2')
gg2 = gain_ifn((ngf*4), 'gg2')
gb2 = bias_ifn((ngf*4), 'gb2')
gw3 = gifn((ngf*4, ngf*2, 5, 5), 'gw3')
gg3 = gain_ifn((ngf*2), 'gg3')
gb3 = bias_ifn((ngf*2), 'gb3')
gw4 = gifn((ngf*2, ngf, 5, 5), 'gw4')
gg4 = gain_ifn((ngf), 'gg4')
gb4 = bias_ifn((ngf), 'gb4')
gwx = gifn((ngf, nc, 5, 5), 'gwx')


aew1 = difn((ndf, nc, 3, 3), 'aew1') 
aew2 = difn((ndf, ndf, 4, 4), 'aew2') 
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


gen_params = [gw, gg, gb, gw2, gg2, gb2, gw3, gg3, gb3, gw4, gg4, gb4, gwx]

discrim_params = [ aew1, aew2, aew3, aew4, aew5, aew6, aeg2, aeb2, aeg3, aeb3, aeg4, aeb4, aeg5, aeb5, aeg6, aeb6, aeg2t, aeb2t, aeg3t, aeb3t, aeg4t, aeb4t, aeg5t, aeb5t, aeg6t, aeb6t]

def gen(Z, w, g, b, w2, g2, b2, w3, g3, b3, w4, g4, b4, wx):
    h = relu(batchnorm(T.dot(Z, w), g=g, b=b))
    h = h.reshape((h.shape[0], ngf*8, 4, 4))
    h2 = relu(batchnorm(deconv(h, w2, subsample=(2, 2), border_mode=(2, 2)), g=g2, b=b2))
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2)), g=g3, b=b3))
    h4 = relu(batchnorm(deconv(h3, w4, subsample=(2, 2), border_mode=(2, 2)), g=g4, b=b4))
    x = tanh(deconv(h4, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x


def discrim(X):
    current_input = dropout(X, 0.3)
    ### encoder ###
    cv1 = relu(dnn_conv(current_input, aew1, subsample=(1,1), border_mode=(1,1)))
    cv2 = relu(batchnorm(dnn_conv(cv1, aew2, subsample=(4,4), border_mode=(0,0)), g=aeg2, b=aeb2))
    cv3 = relu(batchnorm(dnn_conv(cv2, aew3, subsample=(1,1), border_mode=(1,1)), g=aeg3, b=aeb3))
    cv4 = relu(batchnorm(dnn_conv(cv3, aew4, subsample=(4,4), border_mode=(0,0)), g=aeg4, b=aeb4))
    cv5 = relu(batchnorm(dnn_conv(cv4, aew5, subsample=(1,1), border_mode=(1,1)), g=aeg5, b=aeb5))
    cv6 = relu(batchnorm(dnn_conv(cv5, aew6, subsample=(4,4), border_mode=(0,0)), g=aeg6, b=aeb6))

    ### decoder ###
    dv6 = relu(batchnorm(deconv(cv6, aew6, subsample=(4,4), border_mode=(0,0)), g=aeg6t, b=aeb6t)) 
    dv5 = relu(batchnorm(deconv(dv6, aew5, subsample=(1,1), border_mode=(1,1)), g=aeg5t, b=aeb5t))
    dv4 = relu(batchnorm(deconv(dv5, aew4, subsample=(4,4), border_mode=(0,0)), g=aeg4t, b=aeb4t)) 
    dv3 = relu(batchnorm(deconv(dv4, aew3, subsample=(1,1), border_mode=(1,1)), g=aeg3t, b=aeb3t))
    dv2 = relu(batchnorm(deconv(dv3, aew2, subsample=(4,4), border_mode=(0,0)), g=aeg2t, b=aeb2t))
    dv1 = tanh(deconv(dv2, aew1, subsample=(1,1), border_mode=(1,1)))

    rX = dv1
    mse = T.sqrt(T.sum(T.flatten((X-rX)**2, 2), axis=1))
    return (T.flatten(cv6, 2), rX, mse)


X = T.tensor4() # data
X0 = T.tensor4() # vgd samples
X1 = T.tensor4() # vgd samples
deltaX = T.tensor4() #vgd gradient 
Z = T.matrix()


### define discriminative cost ###
_, rX_data, mse_data = discrim(X)
_, rX_vgd, mse_vgd = discrim(X0)
balance_weight = sharedX(0.3)
d_cost = T.mean(mse_data - balance_weight * mse_vgd)

################################# VGD ################################
def vgd_kernel(X0):
    XY = T.dot(X0, X0.transpose())
    x2 = T.reshape(T.sum(T.square(X0), axis=1), (X0.shape[0], 1))
    X2e = T.repeat(x2, X0.shape[0], axis=1)
    H = T.sub(T.add(X2e, X2e.transpose()), 2 * XY)
    
    V = H.flatten()
    
    # median distance
    h = T.switch(T.eq((V.shape[0] % 2), 0),
        # if even vector
        T.mean(T.sort(V)[ ((V.shape[0] // 2) - 1) : ((V.shape[0] // 2) + 1) ]),
        # if odd vector
        T.sort(V)[V.shape[0] // 2])

    h = T.sqrt(0.5 * h / T.log(X0.shape[0].astype('float32') + 1.0)) / 2.

    Kxy = T.exp(-H / h ** 2 / 2.0)
    dxkxy = -T.dot(Kxy, X0)
    sumkxy = T.sum(Kxy, axis=1).dimshuffle(0, 'x')
    dxkxy = T.add(dxkxy, T.mul(X0, sumkxy)) / (h ** 2)
    
    return (Kxy, dxkxy, h)
    

def vgd_gradient(X0, X1):
    # get hidden features
    h1, _, _ = discrim(X1)
    kxy, dxkxy, bw = vgd_kernel(h1) # kernel on hidden features

    # gradient wrt input X0
    h0, _, mse = discrim(X0)

    cost = T.mean(T.sum(T.mul(dxkxy, h0), axis=1))
    dxkxy = T.grad(cost, X0)

    grad = -1.0 * T.grad(T.mean(mse), X0)

    vgd_grad = ( (T.dot(kxy, T.flatten(grad, 2))).reshape(dxkxy.shape) + dxkxy) /  T.sum(kxy, axis=1).reshape((kxy.shape[0],1,1,1))
    return vgd_grad 


gX = gen(Z, *gen_params)
g_cost = -1 * T.sum(T.sum(T.mul(gX, deltaX), axis=1))#update generate models by minimize reconstruct mse


d_lr = 1e-4
g_lr = 1e-3

d_lrt = sharedX(d_lr)
g_lrt = sharedX(g_lr)


d_updater = updates.Adam(lr=d_lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=g_lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))


d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)


print 'COMPILING'
t = time()
_gen = theano.function([Z], gX)
_train_d = theano.function([X, X0], d_cost, updates=d_updates)
_train_g = theano.function([Z, deltaX], g_cost, updates=g_updates)
_vgd_gradient = theano.function([X0, X1], vgd_gradient(X0, X1))
_reconstruction_cost = theano.function([X], T.mean(mse_data))
print '%.2f seconds to compile theano functions'%(time()-t)

sample_zmb = floatX(np_rng.uniform(-1., 1., size=(nvis, nz)))

n_updates = 0
t = time()
for epoch in range(1, niter+1):
    for filename in npzfiles:
        batch_data = shuffle(np.load(filename)['images'].astype(theano.config.floatX))

        for idx in tqdm(xrange(0, batch_data.shape[0] // nbatch)):
            imb = transform(batch_data[idx*nbatch:(idx+1)*nbatch])

            zmb = floatX(np_rng.uniform(-1., 1., size=(imb.shape[0], nz)))
            # generate samples
            samples = _gen(zmb)

            vgd_grad = _vgd_gradient(samples, samples)
            if n_updates % (k+1) == 0:
                _train_g(zmb, floatX(vgd_grad)) 
            else:
                _train_d(imb, samples)
            n_updates += 1

            cost_batch_vgd = _reconstruction_cost(floatX(samples))
            cost_batch_data = _reconstruction_cost(imb)


            if cost_batch_data > cost_batch_vgd:
                d_lrt.set_value(5e-4)
            else:
                d_lrt.set_value(1e-4)

        samples = np.asarray(_gen(sample_zmb))
        color_grid_vis(inverse_transform(samples), (14, 14), 'samples/%s/SteinGAN-%d.png' % (desc, epoch))

    if epoch+1 % 5 == 0:
        joblib.dump([p.get_value() for p in gen_params], 'models/%s/%d_gen_params.jl'%(desc, epoch))
        joblib.dump([p.get_value() for p in discrim_params], 'models/%s/%d_discrim_params.jl'%(desc, epoch))

print '%.2f seconds to train the generative model' % (time()-t)

print 'DONE'
