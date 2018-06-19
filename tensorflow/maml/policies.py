import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype

class MlpPolicy(object):
    """Multi-layer perceptron policy. Policy and Value networks are separated"""
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        """ 
        nbatch : number of samples in a batch
        nsteps : only for RNNs
        """
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        # NN definition. policy and value networks are separated
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, act=lambda x:x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x:x)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        # Create the right probability distribution type depending on the action space (discrete = Categorical, continuous = DiagGaussian)
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

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
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

def construct_weights(dim_input, dim_hidden, dim_output):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal([dim_input, dim_hidden[0]], stddev=0.01), name='w1')
    weights['b1'] = tf.Variable(tf.zeros([dim_hidden[0]]), name='b1')
    for i in range(1,len(dim_hidden)):
        weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([dim_hidden[i-1], dim_hidden[i]], stddev=0.01), name='w'+str(i+1))
        weights['b'+str(i+1)] = tf.Variable(tf.zeros([dim_hidden[i]]), name='b'+str(i+1))
    weights['w'+str(len(dim_hidden)+1)] = tf.Variable(tf.truncated_normal([dim_hidden[-1], dim_output], stddev=0.01), name='w'+str(len(dim_hidden)+1))
    weights['b'+str(len(dim_hidden)+1)] = tf.Variable(tf.zeros([dim_output]), name='b'+str(len(dim_hidden)+1))
    return weights

def normalize(inp, activation, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)

def forward_fc(inp, weights, dim_hidden, reuse=False):
    hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
    for i in range(1,len(dim_hidden)):
        hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights['w'+str(len(dim_hidden)+1)]) + weights['b'+str(len(dim_hidden)+1)]


class MAMLPolicy(object):
    """MAML policy"""
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        """ 
        nbatch : number of samples in a batch
        nsteps : only for RNNs
        """
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        dim_hidden = [100,100]
        # NN definition. policy and value networks are separated
        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("pi"):
                self.pi_weights = construct_weights(dim_input=ob_shape, dim_hidden=dim_hidden, dim_output=actdim)
                pi = forward_fc(X, self.pi_weights, dim_hidden, reuse=reuse)


            with tf.variable_scope("vf"):
                self.vf_weights = construct_weights(dim_input=ob_shape, dim_hidden=dim_hidden, dim_output=1)

            logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())

        # Create the right probability distribution type depending on the action space (discrete = Categorical, continuous = DiagGaussian)
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

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
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


