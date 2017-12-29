import numpy as np
import gym

def get_discounted_returns(rewards, gamma):
    """
    Compute an array of discounted *returns (G_t)*, where each element of the 
    array is the return at that step in the episode.

    This is done by starting at the last step because the value of the 
    last return is simply just the reward at that step.  For all others steps, 
    the return must be computed recursively (based on the next step's return)

    Recall that return is just the sum of all future discounted rewards
    $G_t = r_{t+1} + \gamma*r_{t+2} + \gamma^2*r_{t+3} + ...$
    Return can also be written recursively as:
    $G_i= r_i+1 + \gamma * G_{i+1}$
    
    You should verify this on paper on if you are not sure of it. For more 
    information, a good reference is the 2nd Edition of the Sutton book.
    """
    discounted_returns = np.zeros_like(rewards)

    next_return = 0 # Start next return at 0, since we start at the last action 

    # Start at last value and compute all returns using the recursive definition
    for i in reversed(range(len(rewards))):
        curr_return = rewards[i] + gamma * next_return
        discounted_returns[i] = curr_return

    discounted_returns -= discounted_returns.mean()
    discounted_returns /= discounted_returns.std()
    return discounted_returns

def get_env_io_shapes(env):
    """Return observation_space n, action_space n for an env"""
    ob_n = env.observation_space.shape[0]
    if type(env.action_space) is gym.spaces.discrete.Discrete:
        # For LunarLander discrete
        ac_n = env.action_space.n
    else:
        ac_n = env.action_space.shape[0] 
    return ob_n, ac_n 

