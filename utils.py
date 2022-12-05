import haiku as hk
from haiku import nets
import collections
import random
import numpy as np

Params = collections.namedtuple("Params", "online target")
ActorState = collections.namedtuple("ActorState", "count")
ActorOutput = collections.namedtuple("ActorOutput", "actions q_values")
LearnerState = collections.namedtuple("LearnerState", "count opt_state")
Data = collections.namedtuple("Data", "obs_tm1 a_tm1 r_t discount_t obs_t")


def build_network(num_actions: int, hidden_units) -> hk.Transformed:
  """Factory for a simple MLP network for approximating Q-values."""

  def q(obs):
    network = hk.Sequential(
        [hk.Flatten(),
         nets.MLP([hidden_units, num_actions])])
    return network(obs)

  return hk.without_apply_rng(hk.transform(q))

class ReplayBuffer(object):
  """A simple Python replay buffer."""

  def __init__(self, capacity):
    self._prev = None
    self._action = None
    self._latest = None
    self.buffer = collections.deque(maxlen=capacity)

  def push(self, env_output, action):
    self._prev = self._latest
    self._action = action
    self._latest = env_output

    if action is not None:
      self.buffer.append(
          (self._prev.observation, self._action, self._latest.reward,
           self._latest.discount, self._latest.observation))

  def sample(self, batch_size, discount_factor):
    obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
        *random.sample(self.buffer, batch_size))
    return (np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
            np.asarray(discount_t) * discount_factor, np.stack(obs_t))

  def is_ready(self, batch_size):
    return batch_size <= len(self.buffer)
