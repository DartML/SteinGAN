import time
import numpy as np
nax = np.newaxis
import pdb
DEBUGGER = None

"""
Routines for forward and reverse AIS. The sequence of distributions is given in terms
of a Problem class, which should provide the following methods:
  - init_sample(): return an exact sample from the initial distribution
  - init_partition_function(): compute the exact log partition function of the initial distribution
  - step(state, t): Given the current state, take an MCMC step with inverse temperature t, and return
      the new state
  - joint_prob(state, t): Return the joint unnormalized probability of a state, with inverse temperature t
"""



def sigmoid_schedule(num, rad=4):
    """The sigmoid schedule defined in Section 6.2 of the paper. This is defined as:

          gamma_t = sigma(rad * (2t/T - 1))
          beta_t = (gamma_t - gamma_1) / (gamma_T - gamma_1),

    where sigma is the logistic sigmoid. This schedule allocates more distributions near
    the inverse temperature 0 and 1, since these are often the places where the distributon
    changes the fastest.
    """
    if num == 1:
        return [np.asarray(0.0),np.asarray(1.0)]
    t = np.linspace(-rad, rad, num)
    sigm = 1. / (1. + np.exp(-t))
    return (sigm - sigm.min()) / (sigm.max() - sigm.min())

def LogMeanExp(A,axis=None):
    A_max = np.max(A, axis=axis, keepdims=True)
    B = (
        np.log(np.mean(np.exp(A - A_max), axis=axis, keepdims=True)) +
        A_max
    )
    return B



def ais(problem, schedule):
    """Run AIS in the forward direction. Problem is as defined above, and schedule should
    be an array of monotonically increasing values with schedule[0] == 0 and schedule[-1] == 1."""
    pf = problem.init_partition_function()
    index = 1
    monitor = False

    for it, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:])):
        new_U = problem.eval_lld(t1.astype(np.float32))
        prev_U = problem.eval_lld(t0.astype(np.float32))
        delta = new_U - prev_U
        pf += delta

        accept = problem.step(t1.astype(np.float32))

        if (index+1)% 1000 == 0:
            print "steps %d", index
            print "Accept Prob: %f", np.mean(accept)

        index += 1

    obs_lld = LogMeanExp(pf, axis=1)
    obs_mean = np.mean(obs_lld)


    return obs_mean