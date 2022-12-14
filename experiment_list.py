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
"""Experiment loop."""

import haiku as hk
import jax
import fedjax
import numpy as np

NUM_ROUNDS = 10
def run_loop(
    agent, environment, accumulator, seed,
    batch_size, train_episodes, evaluate_every, eval_episodes, discount_factor):
  """A simple run loop for examples of reinforcement learning with rlax."""

  # Init agent.
  
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed))
  params = agent.initial_params(next(rng))
  learner_state = agent.initial_learner_state(params)
  accuracies_list =[]

  for episode in range(train_episodes):
    accuracies_list.append([])
    timestep = environment.reset()
    accumulator.push(timestep, None)
    actor_state = agent.initial_actor_state()
    print(f"Training agent for {train_episodes} episodes")
    for round in range(NUM_ROUNDS):
      print(f"Episode : {episode}, Round: {round}")
      # Prepare agent, environment and accumulator for a new episode.



      # Acting.
      actor_output, actor_state = agent.actor_step(
          params, timestep, actor_state, next(rng), evaluation=False)

      # Agent-environment interaction.
      action = np.array(actor_output.actions)
      timestep = environment.step(action)

      # Accumulate experience.
      accumulator.push(timestep, action)
      accuracies_list[episode].append(environment.val_acc)
      # Learning.
      # print(accumulator.is_ready(batch_size), "Learning or not")
      # if accumulator.is_ready(batch_size):
      #   print("Using a batch of data")
      #   params, learner_state = agent.learner_step(
      #       params, accumulator.sample(batch_size, discount_factor), learner_state, next(rng))

      params, learner_state = agent.learner_step(params, accumulator.get_last(discount_factor), learner_state, next(rng))
    
      # fedjax.serialization.save_state(server_state.params, '/tmp/params')
      fedjax.serialization.save_state(params, f'/tmp/dqn_params_ckp{episode}')
      fedjax.serialization.save_state(learner_state, f'/tmp/learner_state_ckp{episode}')
      fedjax.serialization.save_state(actor_state, f'/tmp/actor_state_ckp{episode}')

  return params, learner_state, actor_state,accuracies_list
      # Not using the replay buffer
      # params, learner_state = agent.learner_step(params, (), learner_state, next(rng))
    

      # Evaluation. @dhawal, why is this exactly required ?
      # if not episode % evaluate_every:
      #   returns = 0.
      #   for _ in range(eval_episodes):
      #     timestep = environment.reset()
      #     actor_state = agent.initial_actor_state()
      #
      #     tmp = True
      #     while tmp:
      #       actor_output, actor_state = agent.actor_step(
      #           params, timestep, actor_state, next(rng), evaluation=True)
      #       timestep = environment.step(int(actor_output.actions))
      #       returns += timestep.reward
      #       tmp = False
      #
      #   avg_returns = returns / eval_episodes
      #   print(f"Episode {episode:4d}: Average returns: {avg_returns:.2f}")