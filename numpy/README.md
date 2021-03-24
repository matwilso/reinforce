# Basic Policy Gradient methods

## References

- [] http://karpathy.github.io/2016/05/31/rl/
- [] https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 
- [] http://incompleteideas.net/book/the-book-2nd.html
- [] https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/Continuous%20MountainCar%20Actor%20Critic%20Solution.ipynb


## REINFORCE

Thtse train pretty quick.  The variance is moderately high though, so you will 
see a bit of wavering.

```
./reinforce.py --env_id LunarLander-v2 # default
```
to view
```
./reinforce.py --env_id LunarLander-v2 --render_interval 100
```
or 

```
./reinforce.py --env_id Cartpole-v0
```

