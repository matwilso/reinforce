#!/usr/bin/env python3
import argparse
import gym
import numpy as np
import scipy.stats
from itertools import count

# make it possible to import from ../../utils/
import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
from utils.optim import adam

parser = argparse.ArgumentParser(description='Numpy ActorCritic')
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

# TODO: add weight saving and loading?
# TODO: compare the performance of AC vs. reinforce to see if AC is 
# doing something weird different by having a cautious estimate of the
# low value of the ground. Maybe this knowledge make it scared of crashing.
# That would explain why eps take longer.
# ... could also just be a bug

"""
This file implements the Actor-Critic algorithm (or at least a version of it). 
I call it the batched version because this matches very closely to the flow
of the REINFORCE algorithm and waits til the end of the episode to do the 
updates. If you meditated on the code long enough (or checked out the
Sutton book), you would see that it could be doing the update continuously, which is what is done in the other file.

    Resources:
        Sutton and Barto: http://incompleteideas.net/book/the-book-2nd.html
        chapter 13 (read chapters 5 and 6 first on the differences between MC and TD methods. Actor-Critic is the TD equivalent of REINFORCE)


    Glossary:
        (w.r.t.) = with respect to (as in taking gradient with respect to a variable)
        (h or logits) = numerical policy preferences, or unnormalized probailities of actions
"""

class PolicyNetwork(object):
    """
    Neural network policy. Takes in observations and returns probabilities of 
    taking actions.

    ARCHITECTURE:
    {affine - relu } x (L - 1) - affine - softmax  

    """
    def __init__(self, ob_n, ac_n, hidden_dim=500, dtype=np.float32):
        """
        Initialize a neural network to choose actions

        Inputs:
        - ob_n: Length of observation vector
        - ac_n: Number of possible actions
        - hidden_dims: List of size of hidden layer sizes
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.ob_n = ob_n
        self.ac_n = ac_n
        self.hidden_dim = H = hidden_dim
        self.dtype = dtype

        # Initialize all weights (model params) with "Javier Initialization" 
        # weight matrix init = uniform(-1, 1) / sqrt(layer_input)
        # bias init = zeros()
        self.params = {}
        self.params['W1'] = (-1 + 2*np.random.rand(ob_n, H)) / np.sqrt(ob_n)
        self.params['b1'] = np.zeros(H)
        # action head (produce probabilities of taking all actions)
        self.params['W2a'] = (-1 + 2*np.random.rand(H, ac_n)) / np.sqrt(H)
        self.params['b2a'] = np.zeros(ac_n)
        # state-value head (numerical *value* of the state)
        self.params['W2b'] = (-1 + 2*np.random.rand(H, 1)) / np.sqrt(H)
        self.params['b2b'] = np.zeros(1)

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

        # Neural net bookkeeping 
        self.cache = {}
        self.grads = {}
        # Configuration for Adam optimization
        self.optimization_config = {'learning_rate': 1e-3}
        self.adam_configs = {}
        for p in self.params:
            d = {k: v for k, v in self.optimization_config.items()}
            self.adam_configs[p] = d

        # RL specific bookkeeping
        self.saved_action_gradients = []
        self.saved_values = []
        self.rewards = []

    ### HELPER FUNCTIONS
    def _zero_grads(self):
        """Reset gradients to 0. This should be called during optimization steps"""
        for g in self.grads:
            self.grads[g] = np.zeros_like(self.grads[g])

    def _add_to_cache(self, name, val):
        """Helper function to add a parameter to the cache without having to do checks"""
        if name in self.cache:
            self.cache[name].append(val)
        else:
            self.cache[name] = [val]

    def _update_grad(self, name, val):
        """Helper fucntion to set gradient without having to do checks"""
        if name in self.cache:
            self.grads[name] += val
        else:
            self.grads[name] = val

    def _softmax(self, x):
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        return probs

    ### MAIN NEURAL NETWORK STUFF 
    def forward(self, x):
        """
        Forward pass observations (x) through network to get probabilities 
        of taking each action 
        """
        p = self.params
        W1, b1, W2a, b2a, W2b, b2b = p['W1'], p['b1'], p['W2a'], p['b2a'], p['W2b'], p['b2b']

        # forward computations
        affine1 = x.dot(W1) + b1
        relu1 = np.maximum(0, affine1)
        # split the head. one for value estimation, the other for action probs
        affine2a = relu1.dot(W2a) + b2a 
        value = affine2b = relu1.dot(W2b) + b2b 

        logits = affine2a # layer right before softmax (i also call this h)
        # pass through a softmax to get probabilities 
        probs = self._softmax(logits)

        # cache values for backward (based on what is needed for analytic gradient calc)
        self._add_to_cache('affine1', x) 
        self._add_to_cache('relu1', affine1) 
        self._add_to_cache('affine2', relu1) 
        return probs, value
    
    def backward(self, dact, dvalue):
        """
        Backwards pass of the network.
        """
        p = self.params
        W1, b1, W2a, b2a, W2b, b2b = p['W1'], p['b1'], p['W2a'], p['b2a'], p['W2b'], p['W2b']

        # get values from network forward passes (for analytic gradient computations)
        fwd_relu1 = np.concatenate(self.cache['affine2'])
        fwd_affine1 = np.concatenate(self.cache['relu1'])
        fwd_x = np.concatenate(self.cache['affine1'])

        daffine1 = dact.dot(W2a.T) + (dvalue*W2b).T
        # action gradient
        dW2a = fwd_relu1.T.dot(dact)
        db2a = np.sum(dact, axis=0)
        # state value gradient
        dW2b = fwd_relu1.T.dot(dvalue).sum(axis=0) 
        db2b = np.sum(dvalue, axis=0) # note: may be just a scalar

        # gradient of relu (non-negative for values that were above 0 in forward)
        drelu1 = np.where(fwd_affine1 > 0, daffine1, 0)

        # gradient of first affine
        dW1 = fwd_x.T.dot(drelu1)
        db1 = np.sum(drelu1)

        # update gradients 
        self._update_grad('W1', dW1)
        self._update_grad('b1', db1)
        self._update_grad('W2a', dW2a)
        self._update_grad('b2a', db2a)
        self._update_grad('W2b', dW2b)
        self._update_grad('b2b', db2b)

        # reset cache for next backward pass
        self.cache = {}

class ActorCritic(object):
    """
    Object to handle running the algorithm. Uses a PolicyNetwork
    """
    def __init__(self, env):
        ob_n = env.observation_space.shape[0]
        ac_n = env.action_space.n

        self.policy = PolicyNetwork(ob_n, ac_n)

    def select_action(self, obs):
        """
        Pass observations through network and sample an action to take. Keep track
        of dh to use to update weights
        """
        obs = np.reshape(obs, [1, -1])
        netout, value = self.policy.forward(obs)
        netout = netout[0]
        value = value[0]

        std = 0.05 
        probs = netout
        # randomly sample action based on probabilities
        action = np.random.choice(self.policy.ac_n, p=probs)
        # derivative that pulls in direction to make actions taken more probable
        # this will be fed backwards later
        # (see README.md for derivation)
        dh = -1*probs
        dh[action] += 1
        self.policy.saved_action_gradients.append(dh)
        # we save these and we have to wait to calculate the gradient
        # till we have the value of the next state
        # TODO: we could also do this incrementally
        self.policy.saved_values.append(value)
        return action


    def calculate_grads(self, rewards):
        """
        update all at once
        """
        act_grads = np.zeros_like(rewards)
        value_grads = np.zeros_like(rewards)

        discount = 1
        values = self.policy.saved_values

        for t in range(len(rewards)-1):
            td_error = rewards[t] + args.gamma*values[t+1] - values[t]
            value_grads[t] = discount * td_error
            act_grads[t] = discount * td_error
            discount *= args.gamma 

        # why does discount decrease throughout the episode?
        
        last_td = rewards[-1] - values[-1]
        act_grads[-1] = discount * td_error
        value_grads[-1] = discount * td_error
        return act_grads, value_grads
    
    def finish_episode(self):
        """
        At the end of the episode, calculate the discounted return for each time step
        """
        action_gradient = np.array(self.policy.saved_action_gradients)
        act_td_grads, value_td_grads = self.calculate_grads(self.policy.rewards)
        self.policy_gradient = np.zeros(action_gradient.shape)
        self.value_gradient = np.array(value_td_grads)
        for t in range(0, len(act_td_grads)):
            self.policy_gradient[t] = action_gradient[t] * act_td_grads[t]
    
        # negate because we want gradient ascent, not descent
        self.policy.backward(-self.policy_gradient, -self.value_gradient)
    
        # run an optimization step on all of the model parameters
        for p in self.policy.params:
            next_w, self.policy.adam_configs[p] = adam(self.policy.params[p], self.policy.grads[p], config=self.policy.adam_configs[p])
            self.policy.params[p] = next_w
        self.policy._zero_grads() # required every call to adam
    
        # reset stuff
        del self.policy.rewards[:]
        del self.policy.saved_action_gradients[:]
        del self.policy.saved_values[:]


def main():
    """Run ActorCritic algorithm to train on the environment"""
    avg_reward = []
    for i_episode in count(1):
        ep_reward = 0
        obs = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = actor_critic.select_action(obs)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            actor_critic.policy.rewards.append(reward)

            if args.render_interval != -1 and i_episode % args.render_interval == 0:
                env.render()

            if done:
                break

        actor_critic.finish_episode()

        if i_episode % args.log_interval == 0:
            print("Ave reward: {}".format(sum(avg_reward)/len(avg_reward)))
            avg_reward = []

        else:
            avg_reward.append(ep_reward)

if __name__ == '__main__':
    env = gym.make(args.env_id)
    env.seed(args.seed)
    np.random.seed(args.seed)
    actor_critic = ActorCritic(env)
    main()



