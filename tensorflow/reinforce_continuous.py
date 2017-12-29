import numpy as np 
import gym
import itertools
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

H = 400
gamma = 0.99
ACTION_STD = 0.01

def get_env_space_counts(env):
    ob_n = env.observation_space.shape[0]
    if type(env.action_space) is gym.spaces.discrete.Discrete:
        # For LunarLander discrete
        ac_n = 4
    else:
        ac_n = env.action_space.shape[0] 
    return ob_n, ac_n 


def policy_fn(env, features, labels, mode):
    ob_n, ac_n = get_env_space_counts(env)

    input_layer = features["x"]
    dense1 = tf.layers.dense(input_layer, H, activation=tf.nn.relu, name="dense1")
    dense2 = tf.layers.dense(dense1, ac_n, name="dense2")

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

#env = gym.make("BipedalWalker-v2")
#env = gym.make("LunarLander-v2")
#env = gym.make("CarRacing-v0")
#env = gym.make("BipedalWalkerHardcore-v2")
env = gym.make("LunarLanderContinuous-v2")

network_output_cache = []
#action_grad_cache = []
#action_cache = []
reward_cache = []
def sample_policy(tf_obs):
    """
    Sample actions according to the network parameterization, where each of
    output represents the mean of a gaussian to sample an action from (w/ cont std)
    """
    gaussian_means = policy_net(tf_obs)
    action = tf.random_normal(gaussian_means.shape, gaussian_means, ACTION_STD)
    action = tf.clip_by_value(action, -1, 1)

    # Gradient of action based on my interpretation of these slides http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf
    network_output_cache.append(gaussian_means)
    action_cache.appned(action)
    #action_grad_cache.append( (action - gaussian_means) / ACTION_STD )
    return action

#def loss(actions, netouts):
#    return tf.concat(network_output_cache, axis=0)

def end_episode():
    discounted_returns = get_discounted_returns(reward_cache).reshape(-1, 1)
    action_grads = tf.concat(action_grad_cache, axis=0)

    netouts = tf.concat(network_output_cache, axis=0)
    #import ipdb; ipdb.set_trace()
    #optimizer.compute_gradients(loss, actions, netouts)
    optimizer.minimize()

    #del action_grad_cache[:]
    del reward_cache[:]

def run():
    for ep_i in itertools.count(1):
        obs = env.reset()
        tf_obs = tf.constant(obs.astype(np.float32)[np.newaxis, :])
        for t in range(10000): # Don't infinite loop while learning
            action = sample_policy(tf_obs)
            #print(action.numpy()[0,:])

            obs, reward, done, info = env.step(action.numpy()[0,:])
            reward_cache.append(reward)

            #import ipdb; ipdb.set_trace()
            tf_obs = tf.constant(obs.astype(np.float32)[np.newaxis, :])

            if done:
                end_episode()
                break
