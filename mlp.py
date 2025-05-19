import jax
import jax.numpy as jnp
from jax import random, jit
from data_utils import load_mnist
from typing import List, Tuple, Sequence
import tqdm


def init_mlp_params(
  layer_sizes: Sequence[int], key: jax.Array
) -> List[Tuple[jax.Array, jax.Array]]:
  """
  Initialize model weights.
  """
  keys = random.split(key, len(layer_sizes) - 1)
  params = []
  for i, k in enumerate(keys):
    in_dim = layer_sizes[i]
    out_dim = layer_sizes[i + 1]
    stddev = jnp.sqrt(2.0 / in_dim)
    W = random.normal(k, (out_dim, in_dim)) * stddev
    b = jnp.zeros((out_dim,))
    params.append((W, b))
  return params


@jit
def predict(params: List[Tuple[jax.Array, jax.Array]], x: jax.Array) -> jax.Array:
  """
  Forward pass through MLP.
  """
  out = x
  # hidden layers
  for W, b in params[:-1]:
    z = jnp.dot(out, W.T) + b
    out = jax.nn.relu(z)
  # output layer
  W_out, b_out = params[-1]
  logits = jnp.dot(out, W_out.T) + b_out
  log_probs = jax.nn.log_softmax(logits)
  return log_probs


@jit
def loss(
  params: List[Tuple[jax.Array, jax.Array]], x: jax.Array, y_onehot: jax.Array
) -> jax.Array:
  """
  Cross-entropy loss using log-probs from forward pass
  """
  log_probs = predict(params, x)
  xent = -jnp.mean(jnp.sum(log_probs * y_onehot, axis=-1))
  return xent


@jit
def accuracy(
  params: List[Tuple[jax.Array, jax.Array]], x: jax.Array, y: jax.Array
) -> jax.Array:
  """
  Computes classifier accuracy.
  """
  log_probs = predict(params, x)
  preds = jnp.argmax(log_probs, axis=-1)
  acc = jnp.mean(preds == y)
  return acc


@jit
def update(
  params: List[Tuple[jax.Array, jax.Array]],
  x: jax.Array,
  y_onehot: jax.Array,
  lr: float,
) -> List[Tuple[jax.Array, jax.Array]]:
  """
  Perform one SGD update step.
  """
  grads = jax.grad(loss)(params, x, y_onehot)
  updated_params = []
  for (W, b), (dW, db) in zip(params, grads):
    W_updated = W - lr * dW
    b_updated = b - lr * db
    updated_params.append((W_updated, b_updated))
  return updated_params


@jit
def train_step(
  params: List[Tuple[jax.Array, jax.Array]], x: jax.Array, y: jax.Array, lr: float
) -> Tuple[jax.Array, jax.Array]:
  loss_val, grads = jax.value_and_grad(loss)(params, x, y)
  # SGD update
  new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
  return new_params, loss_val


# program entry point
if __name__ == "__main__":
  # load data
  X_train, Y_train, X_test, Y_test = load_mnist("./mnist/")

  # hyperparameters
  layer_sizes = [784, 128, 64, 10]
  learning_rate = 1e-2
  num_epochs = 50
  batch_size = 64

  # initialize model parameters
  key = jax.random.PRNGKey(0)
  params = init_mlp_params(layer_sizes, key)

  # create one-hot encodings for labels
  num_classes = 10
  Y_train_onehot = jax.nn.one_hot(Y_train, num_classes)

  # number of batches
  num_train = X_train.shape[0]
  num_batches = num_train // batch_size

  # training loop
  for epoch in range(num_epochs):
    # shuffle training data
    key, subkey = random.split(key)
    perm = random.permutation(subkey, num_train)
    X_train_shuffled = X_train[perm]
    Y_train_shuffled = Y_train[perm]
    Y_train_onehot_shuffled = Y_train_onehot[perm]

    # mini-batch training
    epoch_loss = 0.0
    for batch in tqdm.trange(num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}"):
      start_idx = batch * batch_size
      end_idx = start_idx + batch_size
      X_batch = X_train_shuffled[start_idx:end_idx]
      Y_batch_onehot = Y_train_onehot_shuffled[start_idx:end_idx]

      # training step: update parameters and compute batch loss
      params, batch_loss = train_step(params, X_batch, Y_batch_onehot, learning_rate)
      epoch_loss += batch_loss / num_batches

    # end-of-epoch evaluation
    train_accuracy = accuracy(params, X_train, Y_train)
    test_accuracy = accuracy(params, X_test, Y_test)
    test_loss = loss(params, X_test, jax.nn.one_hot(Y_test, num_classes))

    print(
      f"Train Loss: {epoch_loss:.4f}, "
      f"Train Accuracy: {train_accuracy * 100:.4f}%, "
      f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )

  # final test set evaluation
  final_test_accuracy = accuracy(params, X_test, Y_test)
  final_test_loss = loss(params, X_test, jax.nn.one_hot(Y_test, num_classes))

  print("\nTraining complete!")
  print(f"Final Test Loss: {final_test_loss:.4f}")
  print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")
