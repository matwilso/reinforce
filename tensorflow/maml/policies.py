import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

### Network construction functions (fc networks and conv networks)
def construct_fc_weights(self):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
    weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
    for i in range(1,len(self.dim_hidden)):
        weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
        weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
    weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
    weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
    return weights

def forward_fc(self, inp, weights, reuse=False):
    hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
    for i in range(1,len(self.dim_hidden)):
        hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]


# TODO: do I want to use batchnorm?


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