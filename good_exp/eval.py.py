import fedjax

env = select_clients.SelectClients(model=model, server_state=server_state, algorithm=algorithm,
                                       train_client_sampler=train_client_sampler, train_fd=train_fd,
                                       num_sampled_clients=10, target_acc=0.99, total_clients=flags_total_participating_clients, seed=flags_seed)
    
# fedjax.serialization.load_state(server_state.params, '/tmp/params')
params = fedjax.serialization.load_state('/tmp/dqn_params')

# fedjax.serialization.load_state( '/tmp/learner_state')
# fedjax.serialization.load_state(actor_state, '/tmp/actor_state')
for round_num in range(1, 1001):
  # Sample 10 clients per round without replacement for training.
  clients = env._all_client_sampler.sample()
#   clients = train_client_sampler.sample()
#   sampled_client_ids = []
#   sampled_client_indices = np.zeros(num_clients, dtype=np.int)
#   for cid, cds, crng in clients:
#     sampled_client_ids.append(cid)
#     sampled_client_indices[all_client_ids.index(cid)] = 1

  # Run one round of training on sampled clients.
  server_state, client_diagnostics = algorithm.apply(server_state, clients)
  print(f'[round {round_num}]')
  # Optionally print client diagnostics if curious about each client's model
  # update's l2 norm.
  # print(f'[round {round_num}] client_diagnostics={client_diagnostics}')

  if round_num % 10 == 0:
    # Periodically evaluate the trained server model parameters.
    # Read and combine clients' train and test datasets for evaluation.
    client_ids = [cid for cid, _, _ in clients]
    train_eval_datasets = [cds for _, cds in train_fd.get_clients(client_ids)]
    test_eval_datasets = [cds for _, cds in test_fd.get_clients(client_ids)]
    train_eval_batches = fedjax.padded_batch_client_datasets(
        train_eval_datasets, batch_size=256)
    test_eval_batches = fedjax.padded_batch_client_datasets(
        test_eval_datasets, batch_size=256)

    # Run evaluation metrics defined in `model.eval_metrics`.
    train_metrics = fedjax.evaluate_model(model, server_state.params,
                                          train_eval_batches)
    test_metrics = fedjax.evaluate_model(model, server_state.params,
                                          test_eval_batches)
    print('[round {round_num}], train_metrics', float(train_metrics['accuracy']))
    print(f'[round {round_num}] test_metrics={test_metrics}')