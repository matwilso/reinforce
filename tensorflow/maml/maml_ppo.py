import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
import baselines
from baselines import logger
from collections import deque
from baselines.common import explained_variance
from baselines.common.runners import AbstractEnvRunner

from .policies import construct_ppo_weights, ppo_forward, ppo_loss

# seed for this code: https://github.com/openai/baselines/tree/master/baselines/ppo2
# paper: https://arxiv.org/abs/1707.06347 

# TODO: major refactor to make everything simpler. after we get it working
# TODO: convert the copy trajs and stuff to some class that I can just change the params I need, like scope name and number in batch

class Model(object):
    """The way this class works is that the __init__ sets up the computational graph and the
    other methods are all for external use, that feed something in to run the desired parts of
    the graph"""
    def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train, nminibatches, 
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

        self.A_A = make_pdtype(ac_space).sample_placeholder([None], name='A_A') # placeholder for sampled action
        self.A_VALUES = tf.placeholder(tf.float32, [None], 'A_VALUES') # Actual values 
        self.A_R = tf.placeholder(tf.float32, [None], 'A_R') # Actual returns 
        self.A_OLDVPRED = tf.placeholder(tf.float32, [None], 'A_OLDVPRED') # Old state-value pred
        self.A_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], 'A_OLDNEGLOGPAC') # Old policy negative log probability (used for prob ratio calc)
        A_TRAJ_DATASET = tf.data.Dataset.from_tensor_slices((self.X, self.A_A, self.A_R, self.A_OLDVPRED, self.A_OLDNEGLOGPAC))
        A_TRAJ_DATASET = A_TRAJ_DATASET.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)
        #A_TRAJ_DATASET = A_TRAJ_DATASET.repeat(noptechos) # may add this back later, after testing
        A_TRAJ_DATASET = A_TRAJ_DATASET.batch(nminibatches) 
        A_TRAJ_ITERATOR = A_TRAJ_DATASET.make_initializable_iterator()

        self.B_A = make_pdtype(ac_space).sample_placeholder([None], name='B_A') # placeholder for sampled action
        self.B_VALUES = tf.placeholder(tf.float32, [None], 'B_VALUES') # Actual values 
        self.B_R = tf.placeholder(tf.float32, [None], 'A_R') # Actual returns 
        self.B_OLDVPRED = tf.placeholder(tf.float32, [None], 'A_OLDVPRED') # Old state-value pred
        self.B_OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], 'A_OLDNEGLOGPAC') # Old policy negative log probability (used for prob ratio calc)
        B_TRAJ_DATSET = tf.data.Dataset.from_tensor_slices((self.X, self.B_A, self.B_R, self.B_OLDVPRED, self.B_OLDNEGLOGPAC))
        B_TRAJ_DATASET = B_TRAJ_DATASET.shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)
        #B_TRAJ_DATASET = B_TRAJ_DATASET.repeat(noptechos)
        B_TRAJ_DATASET = B_TRAJ_DATASET.batch(nminibatches) 
        B_TRAJ_ITERATOR = B_TRAJ_DATSET.make_initializable_iterator()

        # WEIGHTS
        # slow meta weights that only get updated after a full meta-batch
        self.slow_weights = construct_ppo_weights(ob_space, dim_hidden, ac_space, scope='slow')  # type: Dict[str, tf.Variable]
        self.slow_vars = tf.trainable_variables(scope='slow')
        # fast weights used to act in the env
        self.act_weights = construct_ppo_weights(ob_space, dim_hidden, ac_space, scope='act')  # type: Dict[str, tf.Variable]
        self.act_vars = tf.trainable_variables(scope='act')

        # A variable for each of the weights in slow weights, initialize to 0
        # we can pile grads up as in line 10 of the algorithm in MAML Algorithm #3.
        self.meta_grad_pile = {w : tf.Variable(initial_value=tf.zeros_like(self.slow_weights[w]), name='meta_grad_pile_'+w) for w in self.slow_weights}

        # Reset the meta grad 
        self.zero_meta_grad_ops = [tf.assign(self.meta_grad_pile[w], tf.zeros_like(self.meta_grad_pile[w])) for w in self.meta_grad_pile]
        # Sync the slow to the fast 
        self.sync_vars_ops = [tf.assign(act_weight, slow_weight) for act_weight, slow_weight in zip(self.act_vars, self.slow_vars)]

        # ACT
        act_dims = self.processed_x, self.dim_hidden, self.ac_space
        self.act_a, self.act_v, self.act_neglogp, self.act_pd = ppo_forward(act_dims, self.act_weights, scope='act', reuse=False)

        # TRAIN
        # Layout of this section:
        # This is one big line of computation.

        meta_optimizer = tf.train.AdamOptimizer(learning_rate=META_LR, epsilon=1e-5)

        # this part is just for setting up the update on the act weights.
        # (may want to do several epoch runs later here, but idk how maml will perform with that)

        # INNER LOSS
        # TODO: add optimization over noptepochs
        # TODO: seems like I may want to do this in a map_fn like they do in the maml implementation, instead of Dataset

        # run multiple iterations over the inner loss, updating the weights in fast weights
        fast_weights = None 
        for _ in range(nminibatches):
            # 1st iter, we run with self.slow_weights, the rest will be using fast_weights
            weights = fast_weights if fast_weights is not None else self.slow_weights

            A_MB_OBS, A_MB_A, A_MB_VALUES, A_MB_R, A_MB_OLDVPRED, A_MB_OLDNEGLOGPAC = A_TRAJ_ITERATOR.get_next()
            inner_train_dims = A_MB_OBS, self.dim_hidden, self.ac_space
            inner_sample_values = dict(obs=A_MB_OBS, action=A_MB_A, values=A_MB_VALUES, returns=A_MB_R, oldvpred=A_MB_OLDVPREd, oldneglogpac=A_MB_OLDNEGLOGPAC)

            inner_train_a, inner_train_v, inner_train_neglogp, inner_train_pd = ppo_forward(inner_train_dims, weights, scope='act', reuse=True)
            inner_loss = ppo_loss(inner_train_pd, inner_sample_values, self.hyperparams)

            grads = tf.gradients(inner_loss, list(weights.values()))
            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(zip(weights.keys(), [weights[key] - INNER_LR*gradients[key] for key in weights.keys()]))

        # capture the final act weights
        # seems like this is what we would run to update the act weights

        # Run just the inner train op.  The last step of this is to set the act_weights to be the fast weights
        # because we are about to use them to sample another trajectory.
        self.inner_train_op = tf.assign(act_weights, fast_weights)

        # -------------------------------------------------------------------------
        # meta half-way point
        # -------------------------------------------------------------------------
        meta_loss = 0
        for _ in range(nminibatches):
            B_MB_OBS, B_MB_A, B_MB_VALUES, B_MB_R, B_MB_OLDVPRED, B_MB_OLDNEGLOGPAC = B_TRAJ_ITERATOR.get_next()
            meta_train_dims = B_MB_OBS, self.dim_hidden, self.ac_space
            meta_sample_values = dict(obs=B_MB_OBS, action=B_MB_A, values=B_MB_VALUES, returns=B_MB_R, oldvpred=B_MB_OLDVPRED, oldneglogpac=B_MB_OLDNEGLOGPAC)

            # always using the same fast weights for the forward pass
            meta_train_a, meta_train_v, meta_train_neglogp, meta_train_pd = ppo_forward(meta_train_dims, fast_weights, scope='act', reuse=True)
            meta_loss += ppo_loss(meta_train_pd, meta_sample_values, self.hyperparams)


        task_meta_gradients = dict(zip(self.train_weights.keys(), tf.gradients(meta_loss, list(self.train_weights))))
        # add the new task grads in to the meta-batch grad
        self.meta_train_op = update_meta_grad_ops = [tf.assign(self.meta_grad_pile[w], self.meta_grad_pile[w] + task_meta_gradients[w]) for w in self.train_weights]
        # zero out (reset) the meta-batch grad
        zero_meta_grad_pile_ops = [tf.assign(self.meta_grad_pile[w], tf.zeros_like(self.meta_grad_pile[w])) for w in self.meta_grad_pile]

        # zip up the grads to fit the tf.train.Optimizer API, and then apply them to update the slow weights
        meta_grads_and_vars = [(self.meta_grad_pile[w], self.slow_weights[w]) for w in self.slow_weights)]
        self.apply_meta_grad_pile = meta_optimizer.apply_gradients(meta_grads_and_vars, name='meta_grad_step')

    def act(self, obs):
        a, v, neglogp = self.sess.run([self.act_a, self.act_v, self.act_neglogp], {self.X:obs})
        return a, v, neglogp

    def value(self, obs):
        v = self.sess.run([self.act_v], {self.X:obs})
        return v

    def sync_slow_and_fast(self):
        sess.run(self.sync_vars_ops)

    def inner_train(self):
        """inner train on 1 task in the meta-batch"""
        # FEED: traja
        sess.run(self.A_TRAJ_ITERATOR.initializer)
        self.X, self.A_A, self.A_VALUES, self.A_R, self.A_OLDVPRED, self.A_OLDNEGLOGPAC
        feed_dict = {}
        sess.run(self.inner_train_op)

    def meta_train(self):
        """meta train on 1 task in the meta-batch"""
        # these are required to reset the tf.data.Dataset object, 
        # but also resetting the A_TRAJ_ITERATOR here makes sure that you will draw the sample samples from your trajectory
        # and so the gradients will be the same and so your meta gradient will be computing the right thing
        # (i.e. this is very important)
        sess.run(self.A_TRAJ_ITERATOR.initializer) 
        sess.run(self.B_TRAJ_ITERATOR.initializer)

        # sync the fast and slow weights together because we are going to run through all the optimization again
        sess.run(self.sync_vars_ops)

        # list of stuff we need:
        # obs_a, action_a, returns_a, oldvpreds_a, oldneglogpac_a
        # obs_b, action_b, returns_b, oldvpreds_b, oldneglogpac_b
        # lr, cliprange
        sess.run(self.meta_train_op, feed_dict=feed_dict)

    def apply_meta_grad(self):
        """apply the gradient update for the whole meta-batch""""
        sess.run(self.apply_meta_grad_pile)
        sess.run(self.sync_vars_ops) 

class Runner(object):
    """Object to hold RL discounting/trace params and to run a rollout of the policy"""
    def __init__(self, *, env, model, nsteps, gamma, lam, render=False):
        self.env = env
        self.model = model
        self.nsteps = nsteps
        self.gamma = gamma # discount factor
        self.lam = lam # GAE parameter used for exponential weighting of combination of n-step returns
        self.render = render
        self.states = model.initial_state
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(nenv)]
        sess = tf.get_default_session()

        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

    def run(self):
        """Run the policy in env for the set number of steps to collect a trajectory

        Returns: obs, returns, masks, actions, values, neglogpacs, states, epinfos
        """

        # TODO: probably need to randomly draw from the task distrib

        # mb = mini-batch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # Do a rollout of one horizon (not necessarily one ep)
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)            
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            if self.render: 
                self.env.venv.envs[0].render()
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        # discount/bootstrap off value fn (compute advantage)
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0        
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values # I don't get why they do this. Seems only for logging, since they undo it later

        def sf01(arr):
            """swap and then flatten axes 0 and 1"""
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        # obs, returns, dones, actions, values, neglogpacs, states, epinfos
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)), 
            mb_states, epinfos)

def constfn(val):
    def f(_):
        return val
    return f

def meta_learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr, 
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95, 
            log_interval=10, render=False, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0, load_path=None):
    """
    Run training algo for the policy

    policy: policy with step (obs -> act) and value (obs -> v)
    env: (wrapped) OpenAI Gym env
    nsteps: T horizon in PPO paper
    total_timesteps: number of env time steps to take in all
    ent_coef: coefficient for how much to weight entropy in loss
    lr: learning rate. function or float.  function will be passed in progress fraction (t/T) for time adaptive. float will be const
    vf_coef: coefficient for how much to weight value in loss
    max_grad_norm: value for determining how much to clip gradients
    gamma: discount factor
    lam: GAE lambda value (dropoff level for weighting combined n-step rewards. 0 is just 1-step TD estimate. 1 is like value baselined MC)
    nminibathces:  how many mini-batches to split data into (will divide values parameterized by nsteps)
    noptepochs:  how many optimization epochs to run. K in the PPO paper
    cliprange: epsilon in the paper. function or float. see lr for description
    """

    # These allow for time-step adaptive learning rates, where pass in a function that takes in t,
    # but they default to constant functions if you pass in a float
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps # number in the batch
    nbatch_train = nbatch // :minibatches # number in the minibatch for training

    make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, scope='slow_model')

    if save_interval and logger.get_dir():
        import cloudpickle # cloud pickle, because writing a lamdba function (so we can call it later)
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    
    model = make_model()

    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, render=render)



    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates # fraction of num of current update over num total updates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        # collect a trajectory of length nsteps
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() 
        epinfobuf.extend(epinfos)
        mblossvals = []

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)