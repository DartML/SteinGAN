
import theano
import theano.tensor as T
import math
import numpy as np
'''
GMM: Weights M components weights, mu M times d, sigma is a scalar for simplicity. Sigma is variance
'''
class Gau_Ber_RBM:
    def __init__(self, edges, obs_bias, hid_bias, sigma):

        # self.edges = edges.astype(theano.config.floatX)
        # self.obs_bias = obs_bias.astype(theano.config.floatX)
        # self.hid_bias = hid_bias.astype(theano.config.floatX)
        # self.sigma = sigma.astype(theano.config.floatX)

        self.edges = edges
        self.obs_bias = obs_bias
        self.hid_bias = hid_bias
        self.sigma = sigma

    def logp(self, X):
        temp = T.exp(T.dot(X, self.edges) + self.hid_bias.transpose())
        temp_recip = T.exp(-(T.dot(X, self.edges) + self.hid_bias.transpose()))
        logp = T.dot(X, self.obs_bias).dimshuffle(0) - 1.0/(2.0*self.sigma) * T.sum(X*X, axis=1) + T.sum(T.log(temp + temp_recip), axis=1)
        # can be optimalized by log-sum-property
        return logp

    def dlogp(self, X):
        # temp = T.exp(2.0*(T.dot(X, self.edges)+ np.repeat(self.hid_bias.transpose, X.shape[0], axis= 0)))
        temp = T.exp(2.0 * (T.dot(X, self.edges) + self.hid_bias.transpose()))
        dlogp = self.obs_bias.transpose() - 1.0/self.sigma * X + T.dot((temp-1.0)/(temp+1.0), self.edges.transpose())

        return dlogp





