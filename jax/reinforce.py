#!/usr/bin/env python3
import gym
import scipy.stats
import numpy as onp
import jax.numpy as np
from itertools import count
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax, Softmax, softmax
import jax.random as random
from jax.experimental import optimizers
from jax import jit, grad
import jax

"""Something is wrong with this to make it run so slow, but I didn't really want to figure out what at the time so I moved on"""

def one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

def calculate_discounted_returns(rewards):
    returns = onp.zeros(len(rewards))
    next_return = 0 # 0 because we start at the last timestep
    for t in reversed(range(0, len(rewards))):
        next_return = rewards[t] + 0.99 * next_return
        returns[t] = next_return
    # normalize for better statistical properties
    returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps)
    return returns

class REINFORCE(object):
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
    """
    def __init__(self, env):
        net_init, net_apply = stax.serial(
            Dense(128), Relu,
            Dense(128), Relu,
            Dense(4),
            Softmax
        )
        self.key = random.PRNGKey(0)
        self.in_shape = (-1, 8)
        self.out_shape, self.net_params = net_init(self.key, self.in_shape)
        self.apply = lambda inputs: net_apply(self.net_params, inputs)

        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=1e-3)
        self.opt_state = self.opt_init(self.net_params)
        self.opt_t = 1

        def pg_loss(params, sar):
            one_hot_actions = one_hot(sar['a'], 4)
            out = net_apply(params, sar['s'])
            return np.sum(-one_hot_actions * np.log(out))

        @jit
        def _step(i, opt_state, sar):
            params = self.get_params(opt_state)
            g = grad(pg_loss)(params, sar)
            return self.opt_update(i, g, opt_state)

        self.step = _step

     
    def _update_key(self):
        self.key, _ = random.split(self.key)
        return self.key

    def select_action(self, obs):
        obs = np.reshape(obs, [1, -1])
        probs = self.apply(obs)[0]
        uf = random.uniform(self._update_key(), (1,), minval=0.0, maxval=1.0)[0]
        action = np.argmax(uf < np.cumsum(probs))
        return action.item()

    def update(self, sar):
        sar['r'] = calculate_discounted_returns(sar['r'])
        sar['s'] = np.array(sar['s'])
        sar['a'] = np.array(sar['a'])

        self.opt_state = self.step(self.opt_t, self.opt_state, sar)
        self.opt_t += 1
        self.params = self.get_params(self.opt_state)
        



def main():
    """Run REINFORCE algorithm to train on the environment"""

    avg_reward = []
    for i_episode in count(1):
        sar = {key: [] for key in 'sar'}
        obs = env.reset()
        for t in range(10000):  
            action = reinforce.select_action(obs)
            sar['s'].append(obs)
            sar['a'].append(action)
            obs, reward, done, _ = env.step(action)
            sar['r'].append(reward)

            #env.render()

            if done:
                break

        print('1')
        reinforce.update(sar)

        if i_episode % 100 == 0:
            print("Ave reward: {}".format(sum(avg_reward)/len(avg_reward)))
            avg_reward = []
        else:
            avg_reward.append(sum(sar['r']))
    
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    reinforce = REINFORCE(env)
    main()

