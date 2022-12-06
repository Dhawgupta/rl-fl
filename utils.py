import haiku as hk
from haiku import nets
import collections
import random
import numpy as np
import abc
from typing import Iterator, List, Tuple
from fedjax.core import client_datasets
from fedjax.core import federated_data
from fedjax.core.typing import PRNGKey
from fedjax.core.client_samplers import ClientSampler
import jax
import numpy as np
import itertools

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

  def get_last(self, discount_factor, batch_size = 1):
      '''
      Get N last samples
      '''
      obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(*list(itertools.islice(self.buffer, len(self.buffer) - batch_size, len(self.buffer))))
      return (np.stack(obs_tm1), np.asarray(a_tm1), np.asarray(r_t),
              np.asarray(discount_t) * discount_factor, np.stack(obs_t))





def get_pseudo_random_state(seed: int, round_num: int) -> np.random.RandomState:
  """Constructs a deterministic numpy random state."""
  #  Settings for a multiplicative linear congruential generator (aka Lehmer
  #  generator) suggested in 'Random Number Generators: Good Ones are Hard to
  # Find' by Park and Miller.
  mlcg_modulus = 2**(31) - 1
  mlcg_multiplier = 16807
  mlcg_start = np.random.RandomState(seed).randint(1, mlcg_modulus - 1)
  return np.random.RandomState(
      pow(mlcg_multiplier, round_num, mlcg_modulus) * mlcg_start % mlcg_modulus)




class UniformGetClientSampler(ClientSampler):
  """Uniformly samples clients using `FederatedData.get_clients`."""

  def __init__(self,
               fd: federated_data.FederatedData,
               num_clients: int,
               seed: int,
               start_round_num: int = 0,
               custom_client_ids: list = [],
               mal_clients: list = []):
    self._federated_data = fd
    self._num_clients = num_clients
    self._seed = seed
    self._mal_clients = mal_clients
    if not bool(custom_client_ids):
        self._client_ids = list(self._federated_data.client_ids())
    else:
        self._client_ids = custom_client_ids
    self._round_num = start_round_num

  def sample(
      self
  ) -> List[Tuple[federated_data.ClientId, client_datasets.ClientDataset,
                  PRNGKey]]:
    clients = []
    random_state = get_pseudo_random_state(self._seed, self._round_num)
    # Explicitly specify dtype as np.object to prevent numpy from stripping
    # trailing zero bytes from client ids that resulted in lookup KeyErrors.
    client_ids = random_state.choice(
        np.array(self._client_ids, dtype=np.object),
        size=self._num_clients,
        replace=False)
    client_rngs = jax.random.split(
        jax.random.PRNGKey(self._round_num), self._num_clients)
    n_attackers = 0
    for i, (client_id, client_dataset) in enumerate(
        self._federated_data.get_clients(client_ids)):
      if bool(self._mal_clients) and client_id in self._mal_clients:
        n_attackers += 1
        continue
      clients.append((client_id, client_dataset, client_rngs[i]))
    self._round_num += 1
    return clients

  def set_round_num(self, round_num: int):
    self._round_num = round_num











