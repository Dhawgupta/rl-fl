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
flags_train_episodes = 5
# flags.DEFINE_integer("batch_size", 32, "Size of the training batch")
flags_batch_size = 8
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

flags_total_participating_clients = 50

flags_topk = 1



def main():
    # Load train and test federated data for EMNIST.
    train_fd, val_fd = fedjax.datasets.emnist.load_data(only_digits=False)

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
                                       train_client_sampler=train_client_sampler, train_fd=train_fd, val_fd=val_fd,
                                       num_sampled_clients=10, target_acc=0.99, total_clients=flags_total_participating_clients, seed=flags_seed)
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
    params, learner_state, actor_state = experiment.run_loop(
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

    

    # Save final trained model parameters to file.
    # fedjax.serialization.save_state(server_state.params, '/tmp/params')
    # fedjax.serialization.save_state(params, '/tmp/dqn_params')
    # fedjax.serialization.save_state(learner_state, '/tmp/learner_state')
    # fedjax.serialization.save_state(actor_state, '/tmp/actor_state')

    # jnp.save('agent_params', params)
    # jnp.save('learner_state', learner_state)
    # jnp.save('actor_state', actor_state)

    ### EVALUATION BEGINS ### 
    
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)
    
    train_eval_batches = fedjax.padded_batch_client_datasets(
          env._train_eval_datasets, batch_size=256)
    # Run evaluation metrics defined in `model.eval_metrics`.
    train_metrics = fedjax.evaluate_model(env._model, server_state.params,
                                            train_eval_batches)
    print('Intial [round {round_num}], train_metrics', float(train_metrics['accuracy']))
    rl_accuracy_array = [float(train_metrics['accuracy'])]
    for round_num in range(1,101):
      # Sample 10 clients per round without replacement for training.
      clients = env._all_client_sampler.sample()
      clients = clients[:10]
      # Run one round of training on sampled clients.

      env_state = env._create_state_space_server_space_from_server_state(server_state=server_state)
      q_values = agent.actor_step_evaluation(params, env_state, actor_state = None, key = None, evaluation=True, k = 10)
      q_values_sampled_clients = []
      q_values_sampled_clients_idx = []
      for cid, _, _ in clients:
        idx = env._all_client_ids.index(cid)
        q_values_sampled_clients.append(q_values[idx])
        q_values_sampled_clients_idx.append(idx)
      
      sorted_indices = np.flip(np.argsort(q_values_sampled_clients))
      topk_sampled_indices = sorted_indices[:flags_topk]

      accepted_clients = []
      [accepted_clients.append(clients[i][0]) for i in topk_sampled_indices]

      
      
      print(f"accepted_clients: ", accepted_clients)

      
      # indices = np.flip(np.argsort(q_values))
      # topk_idx = indices[:40]
      # print(q_values[topk_idx])
      # print(topk_idx)
      # accepted_clients = []
      # [accepted_clients.append(env._all_client_ids[i]) for i in topk_idx]

      server_state, client_diagnostics = algorithm.apply(server_state, accepted_clients, clients)
      print(f'[round {round_num}]')
      # Optionally print client diagnostics if curious about each client's model
      # update's l2 norm.
      # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

      if round_num % 1 == 0:
        # Periodically evaluate the trained server model parameters.
        # Read and combine clients' train and test datasets for evaluation.
        train_eval_batches = fedjax.padded_batch_client_datasets(
          env._train_eval_datasets, batch_size=256)
        

        # Run evaluation metrics defined in `model.eval_metrics`.
        train_metrics = fedjax.evaluate_model(env._model, server_state.params,
                                                train_eval_batches)
        print('[round {round_num}], train_metrics', float(train_metrics['accuracy']))
        rl_accuracy_array.append(float(train_metrics['accuracy']))
    


    #### RANDOM CODE ####
    
    
    init_params = model.init(jax.random.PRNGKey(17))
    server_state = algorithm.init(init_params)
    
    train_eval_batches = fedjax.padded_batch_client_datasets(
          env._train_eval_datasets, batch_size=256)
    # Run evaluation metrics defined in `model.eval_metrics`.
    train_metrics = fedjax.evaluate_model(env._model, server_state.params,
                                            train_eval_batches)
    print('Intial [round {round_num}], train_metrics', float(train_metrics['accuracy']))
    random_accuracy_array= [float(train_metrics['accuracy'])]
    for round_num in range(1, 101):
      # Sample 10 clients per round without replacement for training.
      clients = env._all_client_sampler.sample()
      # Run one round of training on sampled clients.
      
      
      env_state = env._create_state_space_server_space_from_server_state(server_state=server_state)
    #   q_values = agent.actor_step_evaluation(params, env_state, actor_state = None, key = None, evaluation=True, k = 10)
      indices = np.flip(np.argsort(q_values))
      cls = np.array(list(range(flags_total_participating_clients)))
      np.random.shuffle(cls)
      topk_idx = cls[:flags_topk]
      print(topk_idx)
      accepted_clients = []
      [accepted_clients.append(env._all_client_ids[i]) for i in topk_idx]

      server_state, client_diagnostics = algorithm.apply(server_state, accepted_clients, clients)
      print(f'[round {round_num}]')
      # Optionally print client diagnostics if curious about each client's model
      # update's l2 norm.
      # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

      if round_num % 1 == 0:
        # Periodically evaluate the trained server model parameters.
        # Read and combine clients' train and test datasets for evaluation.
        train_eval_batches = fedjax.padded_batch_client_datasets(
          env._train_eval_datasets, batch_size=256)
        

        # Run evaluation metrics defined in `model.eval_metrics`.
        train_metrics = fedjax.evaluate_model(env._model, server_state.params,
                                                train_eval_batches)
        print('[round {round_num}], train_metrics', float(train_metrics['accuracy']))
        random_accuracy_array.append(float(train_metrics['accuracy']))
    
    rl_accuracy_array = np.array(rl_accuracy_array)
    random_accuracy_array = np.array(random_accuracy_array)
    np.save('rl_array',rl_accuracy_array)
    np.save('random_array',random_accuracy_array)


if __name__ == '__main__':
    # with jax.disable_jit():
    main()