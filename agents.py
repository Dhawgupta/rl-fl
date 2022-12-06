from utils import build_network
import optax
import jax
import jax.numpy as jnp
from utils import Params, ActorOutput, ActorState, LearnerState, Data
import rlax


class DQN:
  """A simple DQN agent."""

  def __init__(self, observation_spec, action_spec, epsilon_cfg, target_period,
               learning_rate, hidden_units):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._target_period = target_period
    # Neural net and optimiser.
    self._network = build_network(action_spec.num_values, hidden_units)
    self._optimizer = optax.adam(learning_rate)
    self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
    # Jitting for speed.
    self.actor_step = jax.jit(self.actor_step)
    self.learner_step = jax.jit(self.learner_step)

  def initial_params(self, key):
    sample_input = self._observation_spec.generate_value()
    sample_input = jnp.expand_dims(sample_input, 0)
    online_params = self._network.init(key, sample_input)
    return Params(online_params, online_params)

  def initial_actor_state(self):
    actor_count = jnp.zeros((), dtype=jnp.float32)
    return ActorState(actor_count)

  def initial_learner_state(self, params):
    learner_count = jnp.zeros((), dtype=jnp.float32)
    opt_state = self._optimizer.init(params.online)
    return LearnerState(learner_count, opt_state)

  def actor_step(self, params, env_output, actor_state, key, evaluation):
    obs = jnp.expand_dims(env_output.observation, 0)  # add dummy batch
    q = self._network.apply(params.online, obs)[0]  # remove dummy batch
    epsilon = self._epsilon_by_frame(actor_state.count)
    train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
    eval_a = rlax.greedy().sample(key, q)
    a = jax.lax.select(evaluation, eval_a, train_a)
    return ActorOutput(actions=a, q_values=q), ActorState(actor_state.count + 1)

  def learner_step(self, params, data, learner_state, unused_key):
    target_params = optax.periodic_update(params.online, params.target,
                                          learner_state.count,
                                          self._target_period)
    dloss_dtheta = jax.grad(self._loss)(params.online, target_params, *data)
    updates, opt_state = self._optimizer.update(dloss_dtheta,
                                                learner_state.opt_state)
    online_params = optax.apply_updates(params.online, updates)
    return (Params(online_params, target_params),
            LearnerState(learner_state.count + 1, opt_state))

  def _loss(self, online_params, target_params, obs_tm1, a_tm1, r_t, discount_t,
            obs_t):
    q_tm1 = self._network.apply(online_params, obs_tm1)
    q_t_val = self._network.apply(target_params, obs_t)
    q_t_select = self._network.apply(online_params, obs_t)
    batched_loss = jax.vmap(rlax.double_q_learning)
    td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
    return jnp.mean(rlax.l2_loss(td_error))


class DQN:
  """A simple DQN agent."""

  def __init__(self, observation_spec, action_spec, epsilon_cfg, target_period,
               learning_rate, hidden_units):
    self._observation_spec = observation_spec
    self._action_spec = action_spec
    self._target_period = target_period
    # Neural net and optimiser.
    self._network = build_network(action_spec.num_values, hidden_units)
    self._optimizer = optax.adam(learning_rate)
    self._epsilon_by_frame = optax.polynomial_schedule(**epsilon_cfg)
    # Jitting for speed.
    self.actor_step = jax.jit(self.actor_step)
    self.learner_step = jax.jit(self.learner_step)

  def initial_params(self, key):
    sample_input = self._observation_spec.generate_value()
    sample_input = jnp.expand_dims(sample_input, 0)
    online_params = self._network.init(key, sample_input)
    return Params(online_params, online_params)

  def initial_actor_state(self):
    actor_count = jnp.zeros((), dtype=jnp.float32)
    return ActorState(actor_count)

  def initial_learner_state(self, params):
    learner_count = jnp.zeros((), dtype=jnp.float32)
    opt_state = self._optimizer.init(params.online)
    return LearnerState(learner_count, opt_state)

  def actor_step(self, params, env_output, actor_state, key, evaluation):
    obs = jnp.expand_dims(env_output.observation, 0)  # add dummy batch
    q = self._network.apply(params.online, obs)[0]  # remove dummy batch
    epsilon = self._epsilon_by_frame(actor_state.count)
    train_a = rlax.epsilon_greedy(epsilon).sample(key, q)
    eval_a = rlax.greedy().sample(key, q)
    a = jax.lax.select(evaluation, eval_a, train_a)
    return ActorOutput(actions=a, q_values=q), ActorState(actor_state.count + 1)

  def learner_step(self, params, data, learner_state, unused_key):
    target_params = optax.periodic_update(params.online, params.target,
                                          learner_state.count,
                                          self._target_period)
    dloss_dtheta = jax.grad(self._loss)(params.online, target_params, *data)
    updates, opt_state = self._optimizer.update(dloss_dtheta,
                                                learner_state.opt_state)
    online_params = optax.apply_updates(params.online, updates)
    return (Params(online_params, target_params),
            LearnerState(learner_state.count + 1, opt_state))

  def _loss(self, online_params, target_params, obs_tm1, a_tm1, r_t, discount_t,
            obs_t):
    q_tm1 = self._network.apply(online_params, obs_tm1)
    q_t_val = self._network.apply(target_params, obs_t)
    q_t_select = self._network.apply(online_params, obs_t)
    batched_loss = jax.vmap(rlax.double_q_learning)
    td_error = batched_loss(q_tm1, a_tm1, r_t, discount_t, q_t_val, q_t_select)
    return jnp.mean(rlax.l2_loss(td_error))