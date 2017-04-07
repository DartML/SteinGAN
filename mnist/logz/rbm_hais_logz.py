import time
import numpy as np
import HAIS_test
from sampler_HAIS import AISPath
from Gauss_Ber_RBM import Gau_Ber_RBM
import scipy.io


def ais_logZ(edges, obs, hid):
     
    #print "AIS iteration"
    obs = np.reshape(obs, (-1, 1))
    hid = np.reshape(hid, (-1, 1))
    #obs = rng.normal(0, 1, (obs_dim, 1)).astype(np.float32)
    #hid = rng.normal(0, 1, (hid_dim, 1)).astype(np.float32)
    #print edges.shape, obs.shape, hid.shape
    
    sigma_rbm = 1.0
    #"Proposal for AIS"
    # mu = np.array([0.0, 0.0]).astype(np.float32)
    # sigma = np.float32(2.0)
    rbmmodel =Gau_Ber_RBM(edges, obs, hid, sigma_rbm)
    
    mu = np.zeros(obs.shape[0])
    sigma = 1.0
    
    #'''Initiliaze parameters'''
    # Trans = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 180, 200, 250 ,300, 350, 400, 450, 500])
    Trans = np.array(
        [1000])
    num_exper = 100
    num_samples = 100
    # epsilon = np.float32(0.01) # starting stepsize
    epsilon = np.float32(0.001)  # starting stepsize
    Leap = 1
    
    path = AISPath(rbmmodel, num_exper, num_samples, mu, sigma, obs.shape[0], epsilon, Leap)
    
    schedule = None
    num_steps = Trans[0]
    schedule = HAIS_test.sigmoid_schedule(num_steps)
    
    logz = HAIS_test.ais(path, schedule)
    
    return logz

#logz_approx = ais_logZ((B_star.astype('float32')), (b_star.astype('float32')), (c_star.astype('float32')))
#print logz_approx
