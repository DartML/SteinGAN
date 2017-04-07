import time
import numpy as np
# from hmc import *
from hmc_single import *
import theano
import theano.tensor as T
import math

np.random.seed(123)
sharedX = (lambda X:
           theano.shared(np.asarray(X, dtype=theano.config.floatX)))


class AISPath:
    def __init__(self, gmmodel, num_exper, num_samples, mu, sigma, hdim, epsilon, Leap):
        '''
        mu and sigma are multivariate gaussian, proposal for AIS, sigma is covariance and scalar
        '''
        self.mu = mu
        self.sigma = sigma
        self.n_sam = num_samples
        self.n_exper = num_exper
        self.gmmodel = gmmodel
        self.L = Leap
        self.eps = epsilon
        self.hdim = hdim
        self.t = T.scalar()
        init_state = self.init_vars()
        self.build(self.eps, self.L, init_state = init_state)

    def init_vars(self):
        init_x = np.zeros((self.n_exper, self.n_sam, self.hdim)) * np.nan
        for i in range(self.n_exper):
            prng = np.random.RandomState(i)
            init_x[i, :, :] = prng.multivariate_normal(self.mu, self.sigma*np.eye(self.hdim), self.n_sam)
        init_x = (np.reshape(init_x, (self.n_exper*self.n_sam, self.hdim))).astype(np.float32)
        init_x = init_x.astype(theano.config.floatX)

        return init_x


    def build(self,
              initial_stepsize,
              n_steps,
              target_acceptance_rate=.65,
              stepsize_dec=0.98,
              stepsize_min=0.0001,
              stepsize_max=0.5,
              stepsize_inc=1.02,
              # used in geometric avg. 1.0 would be not moving at all
              avg_acceptance_slowness=0.9,
              seed=12345,
              init_state=None
              ):

        init_h = init_state

        print ('load init_state')
        # init_m = np.random.randn(self.n_exper * self.n_sam, self.hdim).astype(np.float32)

        # For HMC
        # h denotes current states
        self.h = sharedX(init_h)

        # m denotes momentum
        t = T.scalar()
        # self.generated = self.generate(self.h)
        # lld = T.reshape(-self.energy_fn(self.h), [self.n_exper, self.batch_size])

        lld =  T.reshape(-self.energy_fn(self.h), [self.n_exper, self.n_sam])

        # self.eval_lld = theano.function([t], lld, givens={self.obs: self.obs_val, self.t: t})
        self.eval_lld = theano.function([t], lld, givens={self.t: t})

        # allocate shared variables
        stepsize = sharedX(initial_stepsize)
        avg_acceptance_rate = sharedX(target_acceptance_rate)
        s_rng = TT.shared_randomstreams.RandomStreams(seed)

        # define graph for an `n_steps` HMC simulation
        accept, final_pos = hmc_move(
            s_rng,
            self.h,
            self.energy_fn,
            self.score_energy,
            stepsize,
            n_steps)

        # define the dictionary of updates, to apply on every `simulate` call
        simulate_updates = hmc_updates(
            self.h,
            stepsize,
            avg_acceptance_rate,
            final_pos=final_pos,
            accept=accept,
            stepsize_min=stepsize_min,
            stepsize_max=stepsize_max,
            stepsize_inc=stepsize_inc,
            stepsize_dec=stepsize_dec,
            target_acceptance_rate=target_acceptance_rate,
            avg_acceptance_slowness=avg_acceptance_slowness)

        # self.step = theano.function([t], [accept], updates=simulate_updates, givens={self.obs: self.obs_val, self.t: t})
        self.step = theano.function([t], [accept], updates=simulate_updates, givens={self.t: t})

    def init_partition_function(self):
        return 0.

    def proposal_logpdf(self, state):

        return  - 1.0/2.0 * self.hdim * T.log(2*math.pi*self.sigma) - T.sum((state - self.mu)**2/(2*self.sigma), axis=1)


    def proposal_dlogpdf(self, state):

        return   -(state - self.mu)/float(self.sigma)

    def energy_fn(self, state):

        return - ((1-self.t)*self.proposal_logpdf(state) + self.t * self.gmmodel.logp(state)).astype(theano.config.floatX)

    def score_energy(self, state):

        return - ((1 - self.t)* self.proposal_dlogpdf(state) + self.t * self.gmmodel.dlogp(state)).astype(theano.config.floatX)



# def run_ais(gmmodel, num_exper, num_samples, num_steps, mu, sigma, hdim, epsilon, L, schedule=None):
#     if schedule is None:
#         schedule = ais.sigmoid_schedule(num_steps)
#     path = AISPath(gmmodel, num_exper, num_samples, mu, sigma, hdim, epsilon, L)
#     lld = ais.ais(path, schedule, sigma)
#     return lld


# def run_reverse_ais(model, obs, state, num_steps, sigma, hdim, L, epsilon, data, prior, schedule=None):
#     if schedule is None:
#         schedule = ais.sigmoid_schedule(num_steps)
#     # path = AISPath(model, obs, num_samples, sigma, hdim, L, epsilon,data,prior, init_state = state)
#
#     path = AISPath(model, obs, 1, sigma, hdim, L, epsilon, data, prior, init_state=state)
#     lld = ais.reverse_ais(path, schedule, sigma)
#     return lld
