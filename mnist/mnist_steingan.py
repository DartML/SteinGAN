import sys
sys.path.append('..')

import random
import os
import json
from time import time
import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib

import theano
import theano.tensor as T
from theano.sandbox.cuda.dnn import dnn_conv, dnn_pool
from theano.tensor.shared_randomstreams import RandomStreams

from lib import activations
from lib import updates
from lib import inits
from lib.vis import grayscale_grid_vis
from lib.rng import py_rng, np_rng
from lib.theano_utils import floatX, sharedX
from lib.data_utils import OneHot, shuffle, iter_data
from load import mnist_with_valid_set
import math
from lib.ops import batchnorm, conv_cond_concat, deconv, dropout


# load data first
trX, vaX, teX, trY, vaY, teY = mnist_with_valid_set()
trX, vaX, teX = trX/255., vaX/255., teX/255.
ntrain, nval, ntest = len(trX), len(vaX), len(teX)

k = 1             # # of discrim updates for each gen update
l2 = 1e-5       # l2 weight decay
b1 = 0.5          # momentum term of adam
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
margin = 0.2

def transform(X):
    return (floatX(X)).reshape(-1, nc, npx, npx)

def inverse_transform(X):
    X = X.reshape(-1, npx, npx)
    return X

desc = 'mnist_steingan'

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
bce = T.nnet.binary_crossentropy

gifn = inits.Normal(scale=0.02)
difn = inits.Normal(scale=0.02)
gain_ifn = inits.Normal(loc=1., scale=0.02)
bias_ifn = inits.Constant(c=0.)
dibs = inits.Constant(0.)


gw  = gifn((nz+ny, ngfc), 'gw')
gw2 = gifn((ngfc+ny, ngf*2*7*7), 'gw2')
gw3 = gifn((ngf*2+ny, ngf, 5, 5), 'gw3')
gwx = gifn((ngf+ny, nc, 5, 5), 'gwx')

gen_params = [gw, gw2, gw3, gwx]


def gen(Z, Y, w, w2, w3, wx):
    yb = Y.dimshuffle(0, 1, 'x', 'x')
    Z = T.concatenate([Z, Y], axis=1)
    h = relu(batchnorm(T.dot(Z, w)))
    h = T.concatenate([h, Y], axis=1)
    h2 = relu(batchnorm(T.dot(h, w2)))
    h2 = h2.reshape((h2.shape[0], ngf*2, 7, 7))
    h2 = conv_cond_concat(h2, yb)
    h3 = relu(batchnorm(deconv(h2, w3, subsample=(2, 2), border_mode=(2, 2))))
    h3 = conv_cond_concat(h3, yb)
    x = sigmoid(deconv(h3, wx, subsample=(2, 2), border_mode=(2, 2)))
    return x


aew1 = difn((ndf,nc,5,5), 'aew1')
aew2 = difn((ndf*2,ndf,5,5), 'aew2')
aew3 = difn((ndf*2,ndf*2,3,3), 'aew3')
fw = difn((ndf*2*7*7, 512), 'fw')
aeg2 = gain_ifn((ndf*2), 'aeg2') 
aeb2 = bias_ifn((ndf*2), 'aeb2')
aeg3 = gain_ifn((ndf*2), 'aeg3') 
aeb3 = bias_ifn((ndf*2), 'aeb3')
aeg2t = gain_ifn((ndf), 'aeg2t')
aeb2t = bias_ifn((ndf), 'aeb2t')
aeg3t = gain_ifn((ndf*2), 'aeg3t')
aeb3t = bias_ifn((ndf*2), 'aeb3t')

logistic_w = difn((512,ny), 'logistic_w')
logistic_b = dibs((ny,), 'logistic_b')
 
discrim_params = [aew1, aew2, fw, aeg2, aeb2, aeg2t, aeb2t, logistic_w, logistic_b]


def discrim(X):
    current_input = dropout(X,.3)
    cv1 = relu(dnn_conv(current_input, aew1, subsample=(2,2), border_mode=(2,2)))
    cv2 = relu(batchnorm(dnn_conv(cv1, aew2, subsample=(2,2), border_mode=(2,2)), g=aeg2, b=aeb2))

    fl = T.flatten(cv2, 2)
    hidden = relu( batchnorm(T.dot(fl, fw)))
    flt = relu(batchnorm(T.dot(hidden, fw.T)))

    dv2 = flt.reshape(cv2.shape)
    dv2 = relu(batchnorm(deconv(dv2, aew2, subsample=(2, 2), border_mode=(2,2)), g=aeg2t, b=aeb2t)) 
    dv1 = sigmoid(deconv(dv2, aew1, subsample=(2, 2), border_mode=(2,2)))

    rX = dv1
    mse = T.sqrt(T.sum(T.flatten((X-rX)**2, 2), axis=1))
    return (hidden, rX, mse)


def classifier(H, Y):
    H = dropout(H, 0.3)
    p_y_given_x = T.nnet.softmax(T.dot(H, logistic_w) + logistic_b)
    classification_error = -T.sum(T.mul(T.log(p_y_given_x), Y), axis=1)
    return classification_error


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
    
    return (Kxy, dxkxy)
    


def vgd_gradient(X0, X1, Y):
    # get hidden features
    h1, _, _ = discrim(X1)
    kxy, dxkxy = vgd_kernel(h1) # kernel on hidden features

    # gradient wrt input X0
    h0, _, mse = discrim(X0)
    err = T.maximum(margin, classifier(h0, Y))

    cost = T.mean(T.sum(T.dot(dxkxy, h0.T), axis=1))
    dxkxy = T.grad(cost, X0)

    grad = -1.0 * T.grad(T.mean(mse+err), X0)
    vgd_grad = ((T.dot(kxy, T.flatten(grad, 2))).reshape(dxkxy.shape) + dxkxy) /  T.sum(kxy, axis=1).reshape((kxy.shape[0],1,1,1))
    return vgd_grad



# data
X = T.tensor4()

#  vgd particles 
X0 = T.tensor4()
X1 = T.tensor4()

# vgd gradient 
deltaX = T.tensor4()

# random noise 
Z = T.matrix()

# data label
Y = T.matrix()


### define discriminative cost ###
H_data, reconstruction_data, mse_data = discrim(X)   # data
H_vgd, reconstruction_vgd, mse_vgd = discrim(X0)   # vgd particles 

err_data = T.maximum(margin, classifier(H_data, Y))
err_vgd = T.maximum(margin, classifier(H_vgd, Y))

cost_data = (mse_data + err_data).mean()
cost_vgd = (mse_vgd + err_vgd).mean()

balance_weight = sharedX(0.3)
d_cost = cost_data - balance_weight * cost_vgd   

gX = gen(Z, Y, *gen_params)
g_cost = -1 * T.sum(T.sum(T.mul(gX, deltaX), axis=1)) 


d_lr = 1e-4         # initial learning rate for adam
d_lrt = sharedX(d_lr)

g_lr = 1e-3       # initial learning rate for adam
g_lrt = sharedX(g_lr)

d_updater = updates.Adam(lr=d_lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))
g_updater = updates.Adam(lr=g_lrt, b1=b1, regularizer=updates.Regularizer(l2=l2))

d_updates = d_updater(discrim_params, d_cost)
g_updates = g_updater(gen_params, g_cost)


print 'COMPILING'
t = time()
_train_d = theano.function([X, X0, Y], d_cost, updates=d_updates)
_train_g = theano.function([Z, Y, deltaX], g_cost, updates=g_updates)

_reconstruction_cost = theano.function([X], T.mean(mse_data))

_gen = theano.function([Z, Y], gen(Z, Y, *gen_params))
_discrim = theano.function([X], discrim(X))
_vgd_gradient = theano.function([X0, X1, Y], vgd_gradient(X0, X1, Y))
print '%.2f seconds to compile theano functions'%(time()-t)


sample_zmb = floatX(np_rng.uniform(-1., 1., size=(200, nz)))
sample_ymb = floatX(OneHot(np.asarray([[i for _ in range(20)] for i in range(10)]).flatten(), ny))


print 'training SteinGAN'
n_updates = 0
for epoch in range(niter):
    trX, trY= shuffle(trX, trY)
    # main function
    for imb, ymb in tqdm(iter_data(trX, trY, size=nbatch), total=ntrain/nbatch):
        ymb = floatX(OneHot(ymb, ny))
        zmb = floatX(np_rng.uniform(-1., 1., size=(len(imb), nz)))

        # generate samples
        samples = _gen(zmb, ymb)

        if n_updates % (k+1) == 0:
            _train_g(zmb, ymb, floatX(_vgd_gradient(samples, samples, ymb)))
        else:
            _train_d(transform(imb), floatX(samples), ymb)

        cost_batch_data = _reconstruction_cost(transform(imb))
        cost_batch_vgd = _reconstruction_cost(samples)

        n_updates += 1

        # weight decay
        decay = 1.0 - np.maximum(1.*(epoch-50)/(niter-50), 0.)
        g_lrt.set_value(floatX(g_lr*decay))
        d_lrt.set_value(floatX(d_lr*decay))

        if cost_batch_data > cost_batch_vgd:
            d_lrt.set_value(floatX(5.*d_lrt.get_value()))
            balance_weight.set_value(0.3)
        else:
            balance_weight.set_value(0.1)

        if cost_batch_vgd > cost_batch_data + .5:
            n_updates = n_updates + k+1-(n_updates)%(k+1)
    
    samples = np.asarray(_gen(sample_zmb, sample_ymb))
    grayscale_grid_vis(inverse_transform(samples), (10, 20), '%s/SteinGAN-%d.png' % (samples_dir, epoch))

    if (epoch+1) % 20 == 0:
        print 'dump model...'
        joblib.dump([p.get_value() for p in gen_params], '%s/%d_gen_params.jl'%(model_dir, epoch))
        joblib.dump([p.get_value() for p in discrim_params], '%s/%d_discrim_params.jl'%(model_dir, epoch))

print 'DONE'
