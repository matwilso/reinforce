import numpy as np 
import gym
import itertools
import tensorflow as tf
import tensorflow.contrib as tc
import utils
import argparse

GAMMA=0.99
reward_cache = []
obs_cache = []
action_cache = []

def boolean_flag(parser, name, default=False, help=None):
    dest = name.replace('-', '_')
    parser.add_argument("--" + name, action="store_true", default=default, dest=dest, help=help)
    parser.add_argument("--no-" + name, action="store_false", dest=dest)


class Actor(object):
    """
    dense --([layer_norm] + relu)--> dense --([layer_norm] + relu)--> dense --> out 
    """
    def __init__(self, nb_actions, name='actor', continuous=False):
        self.nb_actions = nb_actions
        self.continuous = continuous
        self.name = name

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = obs
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            if self.continuous:
                x = tf.nn.tanh(x)
            #else:
            #    x = tf.nn.softmax(x)
        return x


class Critic(object):
    """
    dense --([layer_norm] + relu)--> dense --([layer_norm] + relu)--> dense --> out 
    """
    def __init__(self, name='critic'):
        self.name = name

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = obs
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)
            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

def sample_policy(obs):
    actions = sess.run(pred_actions, {obs_tf: obs})[0]
    return actions[0]

def end_episode():
    N = len(reward_cache)
    discounted_returns = utils.get_discounted_returns(reward_cache, GAMMA)

    for t in range(N):
        # prepare inputs
        states  = obs_cache[t][np.newaxis, :]
        actions = np.array([action_cache[t]])
        returns = np.array([discounted_returns[t]])

        # evaluate gradients
        grad_evals = [grad for grad, var in gradients]

        # perform one update of training
        _ = sess.run([
          train_op], feed_dict={
          obs_tf:             states,
          taken_actions:      actions,
          discounted_rewards: returns
        })

    del reward_cache[:]
    del obs_cache[:]
    del action_cache[:]

def run(env_id, seed, layer_norm, evaluation, **kwargs):
    global obs_tf, actor_tf, critic_tf, actions_tf, discounted_rewards, logprobs, cross_entropy_loss, pg_loss, reg_loss, loss, gradients, sess, train_op, optimizer, pred_actions, action_scores, taken_actions

    env = gym.make(env_id)
    ob_n, ac_n = utils.get_env_io_shapes(env)

    critic = Critic()
    actor = Actor(ac_n)
    actions_tf = tf.placeholder(tf.float32, shape=(None,) + (ac_n,), name='actions')
    obs_tf = tf.placeholder(tf.float32, shape=(None,) + (ob_n,), name='obs')
    actor_tf = actor(obs_tf)
    critic_tf = critic(obs_tf, actor_tf)
    action_scores = tf.identity(actor_tf, name="action_scores")
    pred_actions = tf.multinomial(action_scores, 1)

    taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
    discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

    optimizer = tf.train.AdamOptimizer(3e-4)
    logprobs = tf.identity(actor_tf)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logprobs, labels=taken_actions)
    pg_loss            = tf.reduce_mean(cross_entropy_loss)
    #reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
    loss               = pg_loss 
    gradients = optimizer.compute_gradients(loss)
    # compute policy gradients
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (grad * discounted_rewards, var)

    train_op = optimizer.apply_gradients(gradients)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ep_reward = 0
    ep_rewards = []
    

    for ep_i in itertools.count(1):
        obs = env.reset()
        while True:
            action = sample_policy(obs[np.newaxis,:])
            obs, reward, done, info = env.step(action)
            if ep_i % 100 == 0:
                env.render()
            reward_cache.append(reward)
            obs_cache.append(obs)
            action_cache.append(action)
            ep_reward += reward

            #import ipdb; ipdb.set_trace()

            if done:
                end_episode()
                ep_rewards.append(ep_reward)
                ep_reward = 0
                #print(ep_i)
                if ep_i % 100 == 0:
                    avg = sum(ep_rewards) / len(ep_rewards)
                    print("Avg reward = {}".format(avg))
                    ep_rewards = []
                break


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env-id', type=str, default='LunarLander-v2')
    # "LunarLander-v2"
    # "BipedalWalker-v2"
    # "CarRacing-v0"
    # "BipedalWalkerHardcore-v2"
    boolean_flag(parser, 'render-eval', default=False)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'render', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--num-timesteps', type=int, default=None)
    boolean_flag(parser, 'evaluation', default=False)
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args


if __name__ == '__main__':
    args = parse_args()
    run(**args)




