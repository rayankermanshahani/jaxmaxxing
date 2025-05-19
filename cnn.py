import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from data_utils import load_mnist
from typing import Any, Dict, Tuple


def init_cnn_params(key: jax.Array) -> Dict[str, Any]:
  """
  Initialize model parameters for:
  - 4 convolutional layers (conv1, conv2, conv3, conv4)
  - 2 batch normalization layers (bn1, bn2)
  - 1 fully connected layer (fc)

  Args:
    key: JAX random key

  Returns:
    Dictionary containing parameters for each layer.
  """
  keys = random.split(key, 5)
  params = {}

  # first convolution: in=1, out=32, kernel=5x5
  in_channels = 1
  out_channels = 32
  kernel_size = 5
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  params["conv1"] = (
    random.normal(keys[0], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # second convolution: in=32, out=32, kernel=5x5
  in_channels = 32
  out_channels = 32
  kernel_size = 5
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  params["conv2"] = (
    random.normal(keys[1], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # third convolution: in=32, out=64, kernel=3x3
  in_channels = 32
  out_channels = 64
  kernel_size = 3
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  params["conv3"] = (
    random.normal(keys[2], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # fourth convolution: in=64, out=64, kernel=3x3
  in_channels = 64
  out_channels = 64
  kernel_size = 3
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  params["conv4"] = (
    random.normal(keys[3], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # first batch normalization: channels=32
  channels = 32
  params["bn1"] = (
    jnp.ones((channels,)),  # scale (gamma)
    jnp.zeros((channels,)),  # bias (beta)
    jnp.zeros((channels,)),  # running mean (for inference)
    jnp.ones((channels,)),  # running variance (for inference)
  )

  # second batch normalization: channels=64
  channels = 64
  params["bn2"] = (
    jnp.ones((channels,)),  # scale (gamma)
    jnp.zeros((channels,)),  # bias (beta)
    jnp.zeros((channels,)),  # running mean (for inference)
    jnp.ones((channels,)),  # running variance (for inference)
  )

  # fully-connected: in=576, out=10 (576 = 64 channels * 3 * 3 spatial dimensions after pooling)
  in_features = 576
  out_features = 10
  stddev = jnp.sqrt(2.0 / in_features)
  params["fc"] = (
    random.normal(keys[4], (out_features, in_features)) * stddev,
    jnp.zeros((out_features,)),
  )

  return params


@partial(jit, static_argnums=(2,))
def forward(
  params: Dict[str, Any], x: jax.Array, is_training: bool = False
) -> Tuple[jax.Array, Dict[str, Any]]:
  """
  Forward pass through the neural network.

  Args:
    params: Dictionary of model parameters.
    x: Input data [BS x 28 x 28 x 1] in NHWC format or [BS x 784] flattened
    is_training: Toggle between training and inference (affects batch normalization behaviour)

  Returns:
    Tuple of (Output logits [BS x 10], Updated parameters)
  """
  # copy of parameters to update with running batch norm stats during training
  new_params = dict(params) if is_training else params

  # reshape input if it's flat [BS x 784]
  if x.ndim == 2:
    x = x.reshape(-1, 28, 28, 1)

  # first convolution: [BS x 28 x 28 x 1] -> # [BS x 24 x 24 x 32]
  conv1_w, conv1_b = params["conv1"]
  x = jax.nn.relu(
    (
      jax.lax.conv_general_dilated(
        lhs=x,
        rhs=conv1_w,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
      )
      + jnp.reshape(conv1_b, (1, 1, 1, -1))
    )
  )

  # second convolution: [BS x 24 x 24 x 32] -> [BS x 20 x 20 x 32]
  conv2_w, conv2_b = params["conv2"]
  x = jax.nn.relu(
    (
      jax.lax.conv_general_dilated(
        lhs=x,
        rhs=conv2_w,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
      )
      + jnp.reshape(conv2_b, (1, 1, 1, -1))
    )
  )

  # first batch normalization: [BS x 20 x 20 x 32]
  gamma1, beta1, mean1, var1 = params["bn1"]
  if is_training:
    # compute batch stats
    batch_mean = jnp.mean(x, axis=(0, 1, 2))
    batch_var = jnp.var(x, axis=(0, 1, 2))

    # update running stats with momentum
    momentum = 0.9
    new_mean1 = momentum * mean1 + (1 - momentum) * batch_mean
    new_var1 = momentum * var1 + (1 - momentum) * batch_var

    # update parameters with new running stats
    new_params["bn1"] = (gamma1, beta1, new_mean1, new_var1)

    # normalize using batch stats
    batch_mean = batch_mean[None, None, None, :]
    batch_var = batch_var[None, None, None, :]
    x = (x - batch_mean) / jnp.sqrt(batch_var + 1e-5)
  else:  # use stored running stats for inference
    x = (x - mean1[None, None, None, :]) / jnp.sqrt(var1[None, None, None, :] + 1e-5)
  # apply scale and shift
  x = gamma1[None, None, None, :] * x + beta1[None, None, None, :]

  # maxpool: [BS x 20 x 20 x 32] -> [BS x 10 x 10 x 32]
  x = jax.lax.reduce_window(
    operand=x,
    init_value=jnp.inf,
    computation=jax.lax.max,
    window_dimensions=(1, 2, 2, 1),
    window_strides=(1, 2, 2, 1),
    padding="VALID",
  )

  # third convolution: [BS x 10 x 10 x 32] -> [BS x 8 x 8 x 64]
  conv3_w, conv3_b = params["conv3"]
  x = jax.nn.relu(
    (
      jax.lax.conv_general_dilated(
        lhs=x,
        rhs=conv3_w,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
      )
      + conv3_b[None, None, None, :]
    )
  )

  # fourth convolution: [BS x 8 x 8 x 64] -> [BS x 6 x 6 x 64]
  conv4_w, conv4_b = params["conv4"]
  x = jax.nn.relu(
    jax.lax.conv_general_dilated(
      lhs=x,
      rhs=conv4_w,
      window_strides=(1, 1),
      padding="VALID",
      dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    + conv4_b[None, None, None, :]
  )

  # second batch normalization: [BS x 6 x 6 x 64]
  gamma2, beta2, mean2, var2 = params["bn2"]
  if is_training:
    # compute batch stats and update running stats
    batch_mean = jnp.mean(x, axis=(0, 1, 2))
    batch_var = jnp.var(x, axis=(0, 1, 2))

    # update running stats with momentum
    momentum = 0.9
    new_mean2 = momentum * mean2 + (1 - momentum) * batch_mean
    new_var2 = momentum * var2 + (1 - momentum) * batch_var

    # update parameters with new running stats
    new_params["bn2"] = (gamma2, beta2, new_mean2, new_var2)

    # normalize using batch stats
    batch_mean = batch_mean[None, None, None, :]
    batch_var = batch_var[None, None, None, :]
    x = (x - batch_mean) / jnp.sqrt(batch_var + 1e-5)
  else:  # use stored running stats for inference
    x = (x - mean2[None, None, None, :]) / jnp.sqrt(var2[None, None, None, :] + 1e-5)
  # apply scale and shift
  x = gamma2[None, None, None, :] * x + beta2[None, None, None, :]

  # maxpool: [BS x 6 x 6 x 64] -> [BS x 3 x 3 x 64]
  x = jax.lax.reduce_window(
    operand=x,
    init_value=jnp.inf,
    computation=jax.lax.max,
    window_dimensions=(1, 2, 2, 1),
    window_strides=(1, 2, 2, 1),
    padding="VALID",
  )

  # flatten: [BS x 3 x 3 x 64] -> [BS x 576]
  x = x.reshape(x.shape[0], -1)

  # fully connected: [BS x 576] -> [BS x 10]
  fc_w, fc_b = params["fc"]
  x = jnp.dot(x, fc_w.T) + fc_b

  return x, new_params


@jit
def loss(params: Dict[str, Any], x: jax.Array, y_onehot: jax.Array) -> jax.Array:
  """
  Cross entropy-loss function using log-probabilities from forward pass.

  Args:
    params: Dictionary of model parameters.
    x: Input data [BS x 28 x 28 x 1] in NHWC format or [BS x 784] flattened.
    y_onehot: One-hot encoded labels for the input data.

  Returns:
    A JAX Array containing a scalar value of the average loss across a batch.
  """
  log_probs, new_params = forward(params, x, is_training=True)
  cross_entropy_loss = -jnp.mean(jnp.sum(log_probs * y_onehot, axis=-1))
  return cross_entropy_loss


@jit
def accuracy(params: Dict[str, Any], x: jax.Array, y: jax.Array) -> jax.Array:
  """
  Compute the model's accuracy.

  Args:
    params: Dictionary of model parameters.
    x: Input data [BS x 28 x 28 x 1] in NHWC format or [BS x 784] flattened.
    y: Labels for the input data.

  Returns:
    A JAX Array containing a scalar value of the average accuracy across a batch.
  """
  log_probs = forward(params, x, is_training=False)
  preds = jnp.argmax(log_probs, axis=-1)
  acc = jnp.mean(preds == y)
  return acc


@jit
def update():
  """
  TODO:
  Perform one SGD update step.
  """


@jit
def train_step(
  params: Dict[str, Any], x: jax.Array, y: jax.Array, lr: float
) -> Tuple[Dict[str, Any], jax.Array]:
  """
  Perform a single training step.

  Args:
    params: Dictionary of model parameters.
    x: Input data [BS x 28 x 28 x 1] in NHWC format or [BS x 784] flattened.
    y: Labels for the input data.

  Returns:
    A Tuple containing a dictionary of the new parameters and a JAX Array containing the loss value.
  """
  loss_val, grads = jax.value_and_grad(loss)(params, x, y)
  # SGD update
  new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
  return new_params, loss_val


# program entry point
if __name__ == "__main__":
  # load dataset
  X_train, Y_train, X_test, Y_test = load_mnist("./mnist")

  # hyperparameters
  batch_size = 64

  # initialize model parameters
  key = jax.random.PRNGKey(0)
  params = init_cnn_params(key)

  # create one-hot encodings for labels
  num_classes = 10
  Y_train_onehot = jax.nn.one_hot(Y_train, num_classes)

  # number of batches
  num_train = X_train.shape[0]
  num_batches = num_train // batch_size

  # single forward pass
  X_batch = X_train[:batch_size]
  y, new_params = forward(params, X_batch, is_training=True)

  print(f"Forward pass input: {X_batch.shape}, {X_batch.dtype}")
  print(f"Forward pass output: {y.shape}, {y.dtype}")

  print("CNN is complete.")
