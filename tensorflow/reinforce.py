#!/usr/bin/env python3
import argparse
import gym
import numpy as np
import tensorflow as tf
from itertools import count
from collections import namedtuple

parser = argparse.ArgumentParser(description='TensorFlow REINFORCE')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 100)')
parser.add_argument('--render_interval', type=int, default=-1, metavar='N',
                    help='interval between rendering (default: -1)')
parser.add_argument('--env_id', type=str, default='LunarLander-v2',
                    help='gym environment to load')
args = parser.parse_args()

def calculate_discounted_returns(rewards):
    """
    Calculate discounted reward and then normalize it
    (see Sutton book for definition)
    Params:
        rewards: list of rewards for every episode
    """
    returns = np.zeros(len(rewards))

    next_return = 0 # 0 because we start at the last timestep
    for t in reversed(range(0, len(rewards))):
        next_return = rewards[t] + args.gamma * next_return
        returns[t] = next_return
    # normalize for better statistical properties
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    return returns


class PolicyNetworkOpFactory(object):
    def __init__(self, ob_n, ac_n, hidden_dim=500, dtype=tf.float32, name='policy_network'):
        """
        """
        self.ob_n = ob_n
        self.ac_n = ac_n
        self.hidden_dim = H = hidden_dim
        self.dtype = dtype
        self.name = name

    def __call__(self, obs):
        with tf.variable_scope(self.name) as scope:
            x = tf.layers.dense(inputs=obs, units=self.hidden_dim, activation=tf.nn.relu)
            x = tf.layers.dense(inputs=x, units=self.ac_n, activation=None)
        return x


class REINFORCE(object):
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
    """
    def __init__(self, env):
        self.ob_n = env.observation_space.shape[0]
        self.ac_n = env.action_space.n

        self.obs = tf.placeholder(dtype=tf.float32, shape=[None, self.ob_n], name="obs")
        self.action = tf.placeholder(dtype=tf.int32, shape=[None, self.ac_n], name="action")
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="target")
        self.taken_action = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="taken_action")

        self.action_probs = tf.nn.softmax(PolicyNetworkOpFactory(self.ob_n, self.ac_n)(self.obs))
        self.taken_action_probs = tf.gather(self.action_probs, self.taken_action)
        #self.not_taken_probs = ....

        self.loss = -tf.log(self.taken_action_probs) * self.target
        # seems like this only updates the weights w.r.t the chosen action.  all others stay the same.
        # is that broken? probably just less efficient TODO

        # it also looks like we need to pass the observations through the network again to calculate loss
        # also the update looks like is has some bias because you update the weights of the network
        # and the action you might now predict will change, but you are updating based on the old stuff.
        # though maybe the gradient is still pushed in a decent direction
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

    def select_action(self, obs, sess=None):
        """
        Run observation through network and sample an action to take. Keep track
        of dh to use to update weights
        """
        sess = sess or tf.get_default_session()
        probs = sess.run(self.action_probs, {self.obs: obs[None, :]})[0]
        action = np.random.choice(self.ac_n, p=probs)
        return action
    
    def update(self, ep_cache, sess=None):
        returns = calculate_discounted_returns(ep_cache.rewards)
        obs = np.array(ep_cache.obs)
        #taken_actions = np.array(ep_cache.actions)

        sess = sess or tf.get_default_session()
        feed_dict = {self.obs: obs, self.target: returns[:, None]}
        sess.run([self.train_op], feed_dict=feed_dict)

def main():
    """Run REINFORCE algorithm to train on the environment"""

    EpCache = namedtuple("EpCache", ["obs", "actions", "rewards"])
    avg_reward = []
    for i_episode in count(1):
        ep_cache = EpCache([], [], [])
        obs = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = reinforce.select_action(obs)
            obs, reward, done, _ = env.step(action)
            
            ep_cache.obs.append(obs)
            ep_cache.actions.append(action)
            ep_cache.rewards.append(reward)

            if args.render_interval != -1 and i_episode % args.render_interval == 0:
                env.render()

            if done:
                break

        reinforce.update(ep_cache)

        if i_episode % args.log_interval == 0:
            print("Ave reward: {}".format(sum(avg_reward)/len(avg_reward)))
            avg_reward = []
        else:
            avg_reward.append(sum(ep_cache.rewards))

if __name__ == '__main__':
    env = gym.make(args.env_id)
    env.seed(args.seed)
    np.random.seed(args.seed)
    reinforce = REINFORCE(env)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        main()

