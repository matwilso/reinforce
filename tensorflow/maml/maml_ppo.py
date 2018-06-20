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

# seed for this code: https://github.com/openai/baselines/tree/master/baselines/ppo2
# paper: https://arxiv.org/abs/1707.06347 

class Model(object):
    """Object for handling all the network ops, basically losses and values and acting and training and saving"""

    def loss_op(self, ):
        # value prediction
        # do the clipping to prevent too much change/instability in the value function
        vpred = train_model.vf
        vpredclipped = MB_OLDVPRED + tf.clip_by_value(train_model.vf - MB_OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R) 
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2)) # times 0.5 because MSE loss
        # Compute prob ratio between old and new 
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # Clipped Surrogate Objective (max instead of min because values are flipped)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE))) # diagnostic: fraction of values that were clipped
        # total loss = action loss, entropy bonus, and value loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        return loss


    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, scope='model', seed=42):
        sess = tf.get_default_session()
        # model used to draw samples from (nbatch_act is parameterized in case you want to act in multiple envs)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, scope=scope, reuse=False)
        # model used for training the network.  (nbatch_train is the size of minibatch). these share params
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, scope=scope, reuse=True)

        A = train_model.pdtype.sample_placeholder([None]) # placeholder for sampled action
        ADV = tf.placeholder(tf.float32, [None], 'ADV') # Advantage function
        R = tf.placeholder(tf.float32, [None], 'R') # Actual returns 
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], 'OLDNEGLOGPAC') # Old policy negative log probability (used for prob ratio calc)
        OLDVPRED = tf.placeholder(tf.float32, [None], 'OLDVPRED') # Old state-value pred
        LR = tf.placeholder(tf.float32, [], 'LR')
        CLIPRANGE = tf.placeholder(tf.float32, [], 'CLIPRANGE')

        # Dataset for a trajectory
        # This batched optimization needs to happen in the tensorflow graph because it needs to happen in
        # a single run so that ops will be cached.  Because we are optimizing over several epochs I think.
        # with mini-batches I think.
        traj_dataset = tf.data.Dataset.from_tensor_slices((train_model.X, R, A, OLDVPRED, OLDNEGLOGPAC))
        traj_dataset = traj_dataset.shuffle(seed=seed).repeat(noptechos).batch(nminibatch) 
        traj_iterator = traj_dataset.make_initializable_iterator()
        MB_OBS, MB_R, MB_A, MB_OLDVPRED, MB_OLDNEGLOGPAC = traj_iterator.get_next()

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        self.params = params = tf.trainable_variables(scope)

        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm) 

        # TRAINING STUFF
        # TODO: dataset shuffle, need to make sure that it is repeatable


        INDS = tf.placeholder(dtype=tf.int32, shape=[None], name='INDS')
        for K in range(noptepochs):
            shuffled_inds = tf.random_shuffle(INDS, seed=42+K) # must be deterministic to match up between models
            tf.Print(shuffled_inds, [shuffled_inds, "SEED = {}".format(42+K)])
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                mbinds = shuffled_inds[start:end] # mini-batch indices

                slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))  # run training op and collect loss + diagnostics
        
        def train():
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            sess.run()

        # components of loss, and some diagnostics
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            """Dump all trainable variables to a file. (Why use this instead of the TF stuff? Maybe it is better.)"""
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            """Load all trainable variables from a file and load them into TF"""
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        #self.inner_train = inner_train
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step # sample from the policy (feed in obs)
        self.value = act_model.value # get the estimated value of the state
        self.initial_state = act_model.initial_state  
        self.save = save
        self.load = load

class Runner(object):
    """Object to hold RL discounting/trace params and to run a rollout of the policy"""
    def __init__(self, *, env, slow_model, fast_model, nsteps, gamma, lam, render=False):
        self.env = env
        self.slow_model = slow_model
        self.fast_model = fast_model
        self.nsteps = nsteps
        self.gamma = gamma # discount factor
        self.lam = lam # GAE parameter used for exponential weighting of combination of n-step returns
        self.render = render
        self.states = slow_model.initial_state
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=slow_model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(nenv)]
        sess = tf.get_default_session()

        # sync all the variables between the slow and fast models
        sync_ops = [tf.assign(fast_weight, slow_weight) for fast_weight, slow_weight in zip(fast_model.params, slow_model.params)]
        def sync_models():
            sess.run(sync_ops)
        self.sync_models = sync_models

        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

    def run(self, model):
        """Run the policy in env for the set number of steps to collect a trajectory

        Returns: obs, returns, masks, actions, values, neglogpacs, states, epinfos
        """
        model = self.fast_model if model == 'fast' else self.slow_model

        # mb = mini-batch
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        # Do a rollout of one horizon (not necessarily one ep)
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = model.step(self.obs, self.states, self.dones)
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
        last_values = model.value(self.obs, self.states, self.dones)
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
    nbatch_train = nbatch // nminibatches # number in the minibatch for training

    make_slow_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, scope='slow_model')

    make_fast_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, 
                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                    max_grad_norm=max_grad_norm, scope='fast_model')

    if save_interval and logger.get_dir():
        import cloudpickle # cloud pickle, because writing a lamdba function (so we can call it later)
        with open(osp.join(logger.get_dir(), 'make_slow_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_slow_model))
        with open(osp.join(logger.get_dir(), 'make_fast_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_fast_model))
    
    slow_model = make_slow_model()
    fast_model = make_fast_model()
    # WARNING: these datasets must be kept in sync.  they should be drawn from the same number of times

    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, slow_model=slow_model, fast_model=fast_model, nsteps=nsteps, gamma=gamma, lam=lam, render=render)
    runner.sync_models()




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
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run(model='slow') #pylint: disable=E0632
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