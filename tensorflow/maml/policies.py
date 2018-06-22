import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input

# TODO: major refactor to make everything simpler. after we get it working


# GLOSSARY:
# meta = outer = slow = the stuff used to optimize the meta loss
#      = inner = fast = the stuff used to optimize the inner loss.  these get updated more frequently, because this
# makes it easier to sample and optimize the full objective 
 

# GRAPH CHUNKS
# (functions that can be called as part of defining the computational graph)
def batch_norm(inp, activation_fn, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation_fn, reuse=reuse, scope=scope)
def layer_norm(inp, activation_fn, reuse, scope):
    return tf_layers.layer_norm(inp, activation_fn=activation_fn, reuse=reuse, scope=scope)
def construct_fc_weights(dim_input, dim_hidden, dim_output):
    weights = {}
    weights['w1'] = tf.Variable(tf.truncated_normal([dim_input, dim_hidden[0]], stddev=0.01))
    weights['b1'] = tf.Variable(tf.zeros([dim_hidden[0]]))
    for i in range(1,len(dim_hidden)):
        weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([dim_hidden[i-1], dim_hidden[i]], stddev=0.01))
        weights['b'+str(i+1)] = tf.Variable(tf.zeros([dim_hidden[i]]))
    weights['w'+str(len(dim_hidden)+1)] = tf.Variable(tf.truncated_normal([dim_hidden[-1], dim_output], stddev=0.01))
    weights['b'+str(len(dim_hidden)+1)] = tf.Variable(tf.zeros([dim_output]))
    return weights
def forward_fc(inp, weights, dim_hidden, activ=tf.nn.relu, reuse=False):
    hidden = batch_norm(tf.matmul(inp, weights['w1']) + weights['b1'], activation_fn=activ, reuse=reuse, scope='0')
    for i in range(1,len(dim_hidden)):
        hidden = batch_norm(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation_fn=activ, reuse=reuse, scope=str(i+1))
    return tf.matmul(hidden, weights['w'+str(len(dim_hidden)+1)]) + weights['b'+str(len(dim_hidden)+1)]

def ppo_forward(dims, weights, scope, reuse=False):
    obs, dim_hidden, ac_space = dims
    pi_weights, vf_weights, logstd = weights
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope("pi", reuse=reuse):
            pi = forward_fc(obs, pi_weights, dim_hidden=dim_hidden, reuse=reuse)
        with tf.variable_scope("vf", reuse=reuse):
            vf = forward_fc(obs, vf_weights, dim_hidden=dim_hidden, reuse=reuse)
        pdtype = make_pdtype(ac_space)
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        pd = pdtype.pdfromflat(pdparam) # probability distribution

        a0 = pd.sample() # op for sampling from the distribution
        neglogp0 = pd.neglogp(a0) # neglogprob of that action (for gradient)
    return a0, vf, neglogp0, pd

def ppo_loss(pd, sample_values, hyperparams):
    svs = sample_values
    hps = hyperparams

    adv = svs['returns'] - svs['values']
    adv_mean, adv_var = tf.nn.moments(adv, axes=[1])
    adv = (adv - adv_mean) / (adv_var + 1e-8)

    neglogpac = pd.neglogp(svs['action'])
    entropy = tf.reduce_mean(pd.entropy())

    # value prediction
    # do the clipping to prevent too much change/instability in the value function
    vpredclipped = svs['oldvpred'] + tf.clip_by_value(svs['vpred'] - svs['oldvpred'], - hps['cliprange'], hps['cliprange'])
    vf_losses1 = tf.square(svs['vpred'] - svs['returns'])
    vf_losses2 = tf.square(vpredclipped - svs['returns']) 
    vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) # times 0.5 because MSE loss
    # Compute prob ratio between old and new 
    ratio = tf.exp(svs['oldneglogpac'] - neglogpac)
    pg_losses = -adv * ratio
    pg_losses2 = -adv * tf.clip_by_value(ratio, 1.0 - hps['cliprange'], 1.0 + hps['cliprange'])
    # Clipped Surrogate Objective (max instead of min because values are flipped)
    pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
    approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - svs['oldneglogpac']))
    clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), hps['cliprange']))) # diagnostic: fraction of values that were clipped
    # total loss = action loss, entropy bonus, and value loss
    loss = pg_loss - entropy * hps['ent_coef'] + vf_loss * hps['vf_coef']
    return loss


class Model(object):
    def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train, nminibatches
                nsteps, ent_coef, vf_coef, max_grad_norm, dim_hidden=[100,100], scope='model', seed=42):
        """This constructor sets up all the ops and the tensorflow computational graph. The other functions in
        this class are all for sess.runn-ing the various ops"""
        self.sess = tf.get_default_session()
        self.ac_space = ac_space
        self.op_space = ob_space
        self.dim_hidden = dim_hidden
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.X, self.processed_x = observation_input(ob_space)
        self.meta_X, self.meta_processed_x = observation_input(ob_space)
        # TODO: maybe use a different X for acting, to make it a little less confusing and maybe avoid bug

        SHUFFLE_BUFFER_SIZE = nminibatches * nbatch_train
        META_LR = tf.placeholder(tf.float32, [], 'META_LR')
        INNER_LR = tf.placeholder(tf.float32, [], 'INNER_LR')
        CLIPRANGE = tf.placeholder(tf.float32, [], 'CLIPRANGE') # epsilon in the paper
        self.hyperparams = dict(lr=LR, cliprange=CLIPRANGE, ent_coef=ent_coef, vf_coef=vf_coef)

        # TODO: convert this to some class that I can just change the params I need, like scope name and number in batch
        self.A_A = make_pdtype(ac_space).sample_placeholder([None], name='A_A') # placeholder for sampled action
        self.A_VALUES = tf.placeholder(tf.float32, [None], 'A_VALUES') # Actual values 
        self.A_R = tf.placeholder(tf.float32, [None], 'A_R') # Actual returns 
        self.A_OLDVPRED = tf.placeholder(tf.float32, [None], 'A_OLDVPRED') # Old state-value pred
        self.A_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], 'A_OLDNEGLOGPAC') # Old policy negative log probability (used for prob ratio calc)
        A_TRAJ_DATASET = tf.data.Dataset.from_tensor_slices((self.X, self.A_A, self.A_R, self.A_OLDVPRED, self.A_OLDNEGLOGPAC))
        A_TRAJ_DATASET = A_TRAJ_DATASET.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)
        #A_TRAJ_DATASET = A_TRAJ_DATASET.repeat(noptechos)
        A_TRAJ_DATASET = A_TRAJ_DATASET.batch(nminibatch) 
        A_TRAJ_ITERATOR = A_TRAJ_DATASET.make_initializable_iterator()
        #A_MB_OBS, A_MB_A, A_MB_R, A_MB_OLDVPRED, A_MB_OLDNEGLOGPAC = A_TRAJ_ITERATOR.get_next()

        self.B_A = make_pdtype(ac_space).sample_placeholder([None], name='B_A') # placeholder for sampled action
        self.B_VALUES = tf.placeholder(tf.float32, [None], 'B_VALUES') # Actual values 
        self.B_R = tf.placeholder(tf.float32, [None], 'A_R') # Actual returns 
        self.B_OLDVPRED = tf.placeholder(tf.float32, [None], 'A_OLDVPRED') # Old state-value pred
        self.B_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], 'A_OLDNEGLOGPAC') # Old policy negative log probability (used for prob ratio calc)
        B_TRAJ_DATSET = tf.data.Dataset.from_tensor_slices((self.X, self.B_A, self.B_R, self.B_OLDVPRED, self.B_OLDNEGLOGPAC))
        B_TRAJ_DATASET = B_TRAJ_DATASET.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)
        #B_TRAJ_DATASET = B_TRAJ_DATASET.repeat(noptechos)
        B_TRAJ_DATASET = B_TRAJ_DATASET.batch(nminibatch) 
        B_TRAJ_ITERATOR = B_TRAJ_DATSET.make_initializable_iterator()
        #B_MB_OBS, B_MB_A, B_MB_R, B_MB_OLDVPRED, B_MB_OLDNEGLOGPAC = B_TRAJ_ITERATOR.get_next()

        # WEIGHTS
        actdim = ac_space.shape[0]
        def create_weights(scope):
            with tf.variable_scope(scope):
                with tf.variable_scope('pi'):
                    pi_weights = construct_fc_weights(ob_space, dim_hidden, ac_space.shape[0])
                with tf.variable_scope('vf'):
                    vf_weights = construct_fc_weights(ob_space, dim_hidden, 1)
                logstd = tf.get_variable(name="logstd", shape=[1, actdim], initializer=tf.zeros_initializer())
            return pi_weights, vf_weights, logstd
        self.train_weights = self.train_pi_weights, self.train_vf_weights, self.train_logstd = create_weights(scope='train')
        # TODO: make this a mega dictionary
        self.act_weights = self.act_pi_weights, self.act_vf_weights, self.act_logstd = create_weights(scope='act')
        self.train_vars = tf.trainable_variables('train_weights')
        self.act_vars = tf.trainable_variables('act_weights')

        self.sync_vars_ops = [tf.assign(act_weight, train_weight) for act_weight, train_weight in zip(self.act_vars, self.train_vars)]

        # train weights should be called backup weights or something like that. 

        # ACT
        act_dims = self.processed_x, self.dim_hidden, self.ac_space
        self.act_a, self.act_v, self.act_neglogp, self.act_pd = ppo_forward(act_dims, self.act_weights, scope='act', reuse=False)

        # TRAIN
        #inner_optimizer = tf.train.GradientDescentOptimizer(learning_rate=INNER_LR)
        meta_optimizer = tf.train.AdamOptimizer(learning_rate=META_LR, epsilon=1e-5)

        # this part is just for setting up the update on the act weights.
        # (may want to do several epoch runs later here, but idk how maml will perform with that)

        # INNER LOSS
        # TODO: add optimization over noptepochs
        # TODO: seems like I may want to do this in a map_fn like they do in the maml implementation, instead of Dataset

        # run multiple iterations over the inner loss, updating the weights in fast weights
        fast_weights = None 
        for _ in range(nminibatch):
            # 1st iter, we run with self.train_weights, the rest will be using fast_weights
            weights = fast_weights if fast_weights is not None else self.train_weights

            A_MB_OBS, A_MB_A, A_MB_VALUES, A_MB_R, A_MB_OLDVPRED, A_MB_OLDNEGLOGPAC = A_TRAJ_ITERATOR.get_next()
            inner_train_dims = A_MB_OBS, self.dim_hidden, self.ac_space
            inner_sample_values = dict(obs=A_MB_OBS, action=A_MB_A, values=A_MB_VALUES, returns=A_MB_R, oldvpred=A_MB_OLDVPREd, oldneglogpac=A_MB_OLDNEGLOGPAC)

            train_a, train_v, train_neglogp, train_pd = ppo_forward(inner_train_dims, weights, scope='act', reuse=True)
            inner_loss = ppo_loss(train_pd, inner_sample_values, self.hyperparams)

            grads = tf.gradients(inner_loss, list(weights.values()))
            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(zip(weights.keys(), [weights[key] - INNER_LR*gradients[key] for key in weights.keys()]))

        # capture the final act weights
        # seems like this is what we would run to update the act weights

        # Run just the inner train op.  The last step of this is to set the act_weights to be the fast weights
        # because we are about to use them to sample another trajectory.
        inner_train_op = tf.assign(act_weights, fast_weights)

        # -------------------------------------------------------------------------
        # meta half-way point

        meta_loss = 0
        for _ in range(nminibatch):
            B_MB_OBS, B_MB_A, B_MB_VALUES, B_MB_R, B_MB_OLDVPRED, B_MB_OLDNEGLOGPAC = B_TRAJ_ITERATOR.get_next()
            meta_train_dims = B_MB_OBS, self.dim_hidden, self.ac_space
            meta_sample_values = dict(obs=B_MB_OBS, action=B_MB_A, values=B_MB_VALUES, returns=B_MB_R, oldvpred=B_MB_OLDVPRED, oldneglogpac=B_MB_OLDNEGLOGPAC)

            # always using the same fast weights for the forward pass
            train_a, train_v, train_neglogp, train_pd = ppo_forward(meta_train_dims, fast_weights, scope='act', reuse=True)
            meta_loss += ppo_loss(train_pd, meta_sample_values, self.hyperparams)

        # might want to feed in a var list to reduce chance of bug
        meta_train_op = meta_optimizer.minimize(meta_loss)



    def act(self, obs):
        a, v, neglogp = self.sess.run([self.act_a, self.act_v, self.act_neglogp], {self.X:obs})
        return a, v, neglogp

    def value(self, obs):
        v = self.sess.run([self.act_v], {self.X:obs})
        return v

    def sync_slow_and_fast(self):
        sess.run(self.sync_vars_ops)

    def inner_train(self, ):
        # FEED: traja
        self.X, self.A_A, self.A_VALUES, self.A_R, self.A_OLDVPRED, self.A_OLDNEGLOGPAC

        sess.run(inner_train_op)

    def meta_train(self, ):
        # sync the fast and slow weights together because we are going to run through all the optimization again
        sess.run(self.sync_vars_ops)

        # list of stuff we need:
        # obs_a, action_a, returns_a, oldvpreds_a, oldneglogpac_a
        # obs_b, action_b, returns_b, oldvpreds_b, oldneglogpac_b
        # lr, cliprange

        sess.run(meta_train_op, feed_dict=feed_dict)

