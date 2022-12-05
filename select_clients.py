# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""FL client selection reinforcement learning environment."""

from bsuite.environments import base

import dm_env
from dm_env import specs
import numpy as np
import itertools
import fedjax

class SelectClients(base.Environment):
  """A Catch environment built on the dm_env.Environment class.

  The agent must move a paddle to intercept falling balls. Falling balls only
  move downwards on the column they are in.

  The observation is an array shape (rows, columns), with binary values:
  zero if a space is empty; 1 if it contains the paddle or a ball.

  The actions are discrete, and by default there are three available:
  stay, move left, and move right.

  The episode terminates when the ball reaches the bottom of the screen.
  """

  def __init__(self, model, server_state, algorithm, train_client_sampler, train_fd, num_sampled_clients: int = 10, target_acc: float = 1.00, total_clients: int = 3400, seed: int = None):
    """Initializes a new FL client selection environment.

    Args:
      num_sampled_clients: number of sampled clients in each FL round.
      seed: random seed for the RNG.
    """
    self._model = model
    self._server_state = server_state
    self._algorithm = algorithm
    self._num_sampled_clients = num_sampled_clients
    self._train_client_sampler = train_client_sampler
    self._rng = np.random.RandomState(seed)
    self._state_space = None
    self._total_clients = total_clients
    self._obs_space = np.zeros(self._total_clients, dtype=np.float32)
    self._reset_next_step = True
    self._total_regret = 0.
    self._target_acc = target_acc
    self._action_space = np.zeros(self._total_clients, dtype=np.int)
    self._train_fd = train_fd
    self._num_clients = train_fd.num_clients()
    self._all_client_sampler = fedjax.client_samplers.UniformGetClientSampler(fd=train_fd, num_clients=self._num_clients, seed=0)
    self._all_client_ids = []

    for i, client_id in enumerate(itertools.islice(self._train_fd.client_ids(), self._num_clients)):
      self._all_client_ids.append(client_id)
    self._train_eval_datasets = [cds for _, cds in self._train_fd.get_clients(self._all_client_ids)]

  def _reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""
    self._reset_next_step = False
    self._temp = np.random.rand(self._total_clients)
    self._state_space = 2*(self._temp - 0.5)

    return dm_env.restart(self._observation())

  def _step(self, action: int) -> dm_env.TimeStep:
    """Updates the environment according to the action."""
    if self._reset_next_step:
      return self.reset()
    print("Next step")
    # Train only one client with client_index=action

    all_clients = self._all_client_sampler.sample()


    ac = action
    accepted_clients = []
    accepted_clients.append(self._all_client_ids[ac])
    self._server_state, client_diagnostics = self._algorithm.apply(self._server_state, accepted_clients, all_clients)
    
    train_eval_batches = fedjax.padded_batch_client_datasets(
          self._train_eval_datasets, batch_size=256)
    train_metrics = fedjax.evaluate_model(self._model, self._server_state.params,
                                            train_eval_batches)
    print('action= ', action, 'train_accuracy= ', float(train_metrics['accuracy']))

    reward = 2**(float(train_metrics['accuracy']) - self._target_acc)

    # Do you need to change state_space?? 

    # self._state_space = self._state_space - 0.01
    # self._state_space[ac]+0.05
    # self._state_space = np.clip(self._state_space, -1, 1)

    return dm_env.transition(reward=reward, observation=self._observation())

  def observation_spec(self) -> specs.BoundedArray:
    """Returns the observation spec."""
    return specs.BoundedArray(shape=self._obs_space.shape, dtype=self._obs_space.dtype,
                              name="board", minimum=-1, maximum=1)

  def action_spec(self) -> specs.DiscreteArray:
    """Returns the action spec."""
    return specs.DiscreteArray(
        dtype=np.int, num_values=int(self._total_clients), name="action")

  def _observation(self) -> np.ndarray:
    return self._state_space.copy()

  def bsuite_info(self):
    return dict(total_regret=self._total_regret)
