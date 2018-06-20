import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

class MAMLPolicy(object):
    """MAML policy"""
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, scope='model', reuse=False): #pylint: disable=W0613
        """ 
        nbatch : number of samples in a batch
        nsteps : only for RNNs
        """
        # Create the right probability distribution type depending on the action space (discrete = Categorical, continuous = DiagGaussian)
        self.pdtype = make_pdtype(ac_space)
        # NN definition. policy and value networks are separated
        with tf.variable_scope(scope, reuse=reuse):
            X, processed_x = observation_input(ob_space, nbatch)
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample() # op for sampling from the distribution
        neglogp0 = self.pd.neglogp(a0) # neglogprob of that action (for gradient)
        self.initial_state = None # just for RNNs

        def step(ob, *_args, **_kwargs):
            """Feed the observation through network to get action, value, and neglogprob"""
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            """Feed the observation through to get the value"""
            return sess.run(vf, {X:ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value