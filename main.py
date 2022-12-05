# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple double-DQN agent trained to play BSuite's Catch env."""

import collections
import random
from absl import app
from absl import flags
from bsuite.environments import catch
import select_clients
import haiku as hk
from haiku import nets
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
import experiment
# from rlax.rlax._src.distributions import epsilon_greedy
import fedjax
import fed_avg
from utils import build_network, ReplayBuffer
from utils import Params, ActorState, ActorOutput, LearnerState, Data
import itertools
from agents import DQN

# import sys
# sys.argv = sys.argv[:1]


FLAGS = flags.FLAGS

# flags.DEFINE_integer("seed", 42, "Random seed.")
flags_seed=42
# flags.DEFINE_integer("train_episodes", 301, "Number of train episodes.")
flags_train_episodes = 1001
# flags.DEFINE_integer("batch_size", 32, "Size of the training batch")
flags_batch_size = 32
# flags.DEFINE_float("target_period", 50, "How often to update the target net.")
flags_target_period = 50
# flags.DEFINE_integer("replay_capacity", 2000, "Capacity of the replay buffer.")
flags_replay_capacity = 2000
# flags.DEFINE_integer("hidden_units", 50, "Number of network hidden units.")
flags_hidden_units = 50
# flags.DEFINE_float("epsilon_begin", 1., "Initial epsilon-greedy exploration.")
flags_epsilon_begin = 1
# flags.DEFINE_float("epsilon_end", 0.01, "Final epsilon-greedy exploration.")
flags_epsilon_end = 0.01
# flags.DEFINE_integer("epsilon_steps", 1000, "Steps over which to anneal eps.")
flags_epsilon_steps = 1000
# flags.DEFINE_float("discount_factor", 0.99, "Q-learning discount factor.")
flags_discount_factor = 0.99
# flags.DEFINE_float("learning_rate", 0.005, "Optimizer learning rate.")
flags_learning_rate = 0.005
# flags.DEFINE_integer("eval_episodes", 100, "Number of evaluation episodes.")
flags_eval_episodes = 100
# flags.DEFINE_integer("evaluate_every", 50,
#                      "Number of episodes between evaluations.")
flags_evaluate_every = 1



def main():
    # Load train and test federated data for EMNIST.
    train_fd, test_fd = fedjax.datasets.emnist.load_data(only_digits=False)

    # Create CNN model with dropout.
    model = fedjax.models.emnist.create_conv_model(only_digits=False)

    # Scalar loss function with model parameters, batch of examples, and seed
    # PRNGKey as input.
    def loss(params, batch, rng):
        # `rng` used with `apply_for_train` to apply dropout during training.
        preds = model.apply_for_train(params, batch, rng)
        # Per example loss of shape [batch_size].
        example_loss = model.train_loss(batch, preds)
        return jnp.mean(example_loss)

    # Gradient function of `loss` w.r.t. to model `params` (jitted for speed).
    grad_fn = jax.jit(jax.grad(loss))

    # Create federated averaging algorithm.
    client_optimizer = fedjax.optimizers.sgd(learning_rate=10 ** (-1.5))
    server_optimizer = fedjax.optimizers.adam(
        learning_rate=10 ** (-2.5), b1=0.9, b2=0.999, eps=10 ** (-4))
    # Hyperparameters for client local traing dataset preparation.
    client_batch_hparams = fedjax.ShuffleRepeatBatchHParams(batch_size=20)
    algorithm = fed_avg.federated_averaging(grad_fn, client_optimizer,
                                            server_optimizer,
                                            client_batch_hparams)

    # Initialize model parameters and algorithm server state.
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)

    # Train and eval loop.
    train_client_sampler = fedjax.client_samplers.UniformGetClientSampler(
        fd=train_fd, num_clients=10, seed=0)

    num_clients = train_fd.num_clients()
    all_client_ids = []
    for i, client_id in enumerate(itertools.islice(train_fd.client_ids(), num_clients)):
        all_client_ids.append(client_id)

    ############################  RL CODE HERE  ##################################
    env = select_clients.SelectClients(model=model, server_state=server_state, algorithm=algorithm,
                                       train_client_sampler=train_client_sampler, train_fd=train_fd,
                                       num_sampled_clients=10, target_acc=0.99, total_clients=3400, seed=flags_seed)
    epsilon_cfg = dict(
        init_value=flags_epsilon_begin,
        end_value=flags_epsilon_end,
        transition_steps=flags_epsilon_steps,
        power=1.)
    agent = DQN(
        observation_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        epsilon_cfg=epsilon_cfg,
        target_period=flags_target_period,
        learning_rate=flags_learning_rate,
        hidden_units=flags_hidden_units,
    )

    accumulator = ReplayBuffer(flags_replay_capacity)
    experiment.run_loop(
        agent=agent,
        environment=env,
        accumulator=accumulator,
        seed=flags_seed,
        batch_size=flags_batch_size,
        train_episodes=flags_train_episodes,
        evaluate_every=flags_evaluate_every,
        eval_episodes=flags_eval_episodes,
        discount_factor=flags_discount_factor,
    )
    ##############################################################################

    # for round_num in range(1, 1501):
    #   # Sample 10 clients per round without replacement for training.
    #   clients = train_client_sampler.sample()
    #   sampled_client_ids = []
    #   sampled_client_indices = np.zeros(num_clients, dtype=np.int)
    #   for cid, cds, crng in clients:
    #     sampled_client_ids.append(cid)
    #     sampled_client_indices[all_client_ids.index(cid)] = 1

    #   # Run one round of training on sampled clients.
    #   server_state, client_diagnostics = algorithm.apply(server_state, clients)
    #   print(f'[round {round_num}]')
    #   # Optionally print client diagnostics if curious about each client's model
    #   # update's l2 norm.
    #   # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

    #   if round_num % 10 == 0:
    #     # Periodically evaluate the trained server model parameters.
    #     # Read and combine clients' train and test datasets for evaluation.
    #     client_ids = [cid for cid, _, _ in clients]
    #     train_eval_datasets = [cds for _, cds in train_fd.get_clients(client_ids)]
    #     test_eval_datasets = [cds for _, cds in test_fd.get_clients(client_ids)]
    #     train_eval_batches = fedjax.padded_batch_client_datasets(
    #         train_eval_datasets, batch_size=256)
    #     test_eval_batches = fedjax.padded_batch_client_datasets(
    #         test_eval_datasets, batch_size=256)

    #     # Run evaluation metrics defined in `model.eval_metrics`.
    #     train_metrics = fedjax.evaluate_model(model, server_state.params,
    #                                           train_eval_batches)
    #     test_metrics = fedjax.evaluate_model(model, server_state.params,
    #                                          test_eval_batches)
    #     print('[round {round_num}], train_metrics', float(train_metrics['accuracy']))
    #     print(f'[round {round_num}] test_metrics={test_metrics}')

    # Save final trained model parameters to file.
    fedjax.serialization.save_state(server_state.params, '/tmp/params')


if __name__ == '__main__':
    main()