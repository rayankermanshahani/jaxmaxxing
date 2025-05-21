import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from data_utils import load_mnist
from typing import Tuple, NamedTuple
import tqdm


class BatchNormState(NamedTuple):
  """
  Track running stats for batch normalization.
  """

  mean1: jax.Array  # running mean for first batchnorm layer
  var1: jax.Array  # running variance for first batchnorm layer
  mean2: jax.Array  # running mean for second batchnorm layer
  var2: jax.Array  # running variance for second batchnorm layer


class ModelParams(NamedTuple):
  """
  Contains all trainable parameters of the model.
  """

  conv1: Tuple[jax.Array, jax.Array]  # (weights, bias)
  conv2: Tuple[jax.Array, jax.Array]  # (weights, bias)
  conv3: Tuple[jax.Array, jax.Array]  # (weights, bias)
  conv4: Tuple[jax.Array, jax.Array]  # (weights, bias)
  bn1: Tuple[jax.Array, jax.Array]  # (gamma, beta)
  bn2: Tuple[jax.Array, jax.Array]  # (gamma, beta)
  fc: Tuple[jax.Array, jax.Array]  # (weights, bias)


def init_model() -> Tuple[ModelParams, BatchNormState]:
  """
  Initialize model parameters and batch normalization state.

  Returns:
    A tuple of (trainable_parameters, batch norm state).
  """
  key = jax.random.PRNGKey(0)
  keys = jax.random.split(key, 5)
  params = {}

  # first convolution: in=1, out=32, kernel=5x5
  in_channels = 1
  out_channels = 32
  kernel_size = 5
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  conv1 = (
    jax.random.normal(keys[0], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # second convolution: in=32, out=32, kernel=5x5
  in_channels = 32
  out_channels = 32
  kernel_size = 5
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  conv2 = (
    jax.random.normal(keys[1], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # third convolution: in=32, out=64, kernel=3x3
  in_channels = 32
  out_channels = 64
  kernel_size = 3
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  conv3 = (
    jax.random.normal(keys[2], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # fourth convolution: in=64, out=64, kernel=3x3
  in_channels = 64
  out_channels = 64
  kernel_size = 3
  stddev = jnp.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
  conv4 = (
    jax.random.normal(keys[3], (out_channels, in_channels, kernel_size, kernel_size))
    * stddev,
    jnp.zeros((out_channels,)),
  )

  # first batch normalization: channels=32
  channels = 32
  bn1 = (
    jnp.ones((channels,)),  # scale (gamma)
    jnp.zeros((channels,)),  # bias (beta)
  )

  # second batch normalization: channels=64
  channels = 64
  bn2 = (
    jnp.ones((channels,)),  # scale (gamma)
    jnp.zeros((channels,)),  # bias (beta)
  )

  # fully-connected: in=576, out=10 (576 = 64 channels * 3 * 3 spatial dimensions after pooling)
  in_features = 576
  out_features = 10
  stddev = jnp.sqrt(2.0 / in_features)
  fc = (
    jax.random.normal(keys[4], (out_features, in_features)) * stddev,
    jnp.zeros((out_features,)),
  )

  # create model parameters
  params = ModelParams(
    conv1=conv1,
    conv2=conv2,
    conv3=conv3,
    conv4=conv4,
    bn1=bn1,
    bn2=bn2,
    fc=fc,
  )

  # create batch norm running stats
  bn_state = BatchNormState(
    mean1=jnp.zeros((32,)),
    var1=jnp.ones((32,)),
    mean2=jnp.zeros((64,)),
    var2=jnp.ones((64,)),
  )

  return params, bn_state


@partial(jit, static_argnums=(3, 4))
def forward(
  params: ModelParams,
  bn_state: BatchNormState,
  x: jax.Array,
  is_training: bool = False,
  momentum: float = 0.9,
) -> Tuple[jax.Array, BatchNormState]:
  """
  Forward pass through the neural network.

  Args:
    params: Dictionary of model parameters.
    bn_state: Current batch normalization layers' state.
    x: Input data [BS x 28 x 28 x 1] in NHWC format or [BS x 784] flattened.
    is_training: Toggle between training and inference (affects batch normalization behaviour).
    momentum: Momentum for updating running statistics (only used in training mode).

  Returns:
    Tuple of (output logits [BS x 10], batch normalization state)
  """
  # reshape input if it's flat [BS x 784]
  if x.ndim == 2:
    x = x.reshape(-1, 28, 28, 1)

  # variables to store updated batchnorm stats if training
  new_mean1, new_var1 = None, None
  new_mean2, new_var2 = None, None

  # first convolution: [BS x 28 x 28 x 1] -> # [BS x 24 x 24 x 32]
  conv1_w, conv1_b = params.conv1
  x = jax.nn.relu(
    (
      jax.lax.conv_general_dilated(
        lhs=x,
        rhs=conv1_w,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
      )
      + jnp.reshape(conv1_b, (1, 1, 1, -1))  # reshaping for broadcasting
    )
  )

  # second convolution: [BS x 24 x 24 x 32] -> [BS x 20 x 20 x 32]
  conv2_w, conv2_b = params.conv2
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
  gamma1, beta1 = params.bn1
  if is_training:
    # compute batch stats
    batch_mean1 = jnp.mean(x, axis=(0, 1, 2))
    batch_var1 = jnp.var(x, axis=(0, 1, 2))

    # normalize using batch stats
    x = (x - jnp.reshape(batch_mean1, (1, 1, 1, -1))) / jnp.sqrt(
      jnp.reshape(batch_var1, (1, 1, 1, -1)) + 1e-5
    )

    # update running stats
    new_mean1 = momentum * bn_state.mean1 + (1 - momentum) * batch_mean1
    new_var1 = momentum * bn_state.var1 + (1 - momentum) * batch_var1

  else:
    # normalize using running stats
    x = (x - jnp.reshape(bn_state.mean1, (1, 1, 1, -1))) / jnp.sqrt(
      jnp.reshape(bn_state.var1, (1, 1, 1, -1)) + 1e-5
    )
  # apply scale and shift
  x = jnp.reshape(gamma1, (1, 1, 1, -1)) * x + jnp.reshape(beta1, (1, 1, 1, -1))

  # maxpool: [BS x 20 x 20 x 32] -> [BS x 10 x 10 x 32]
  x = jax.lax.reduce_window(
    operand=x,
    init_value=-jnp.inf,
    computation=jax.lax.max,
    window_dimensions=(1, 2, 2, 1),
    window_strides=(1, 2, 2, 1),
    padding="VALID",
  )

  # third convolution: [BS x 10 x 10 x 32] -> [BS x 8 x 8 x 64]
  conv3_w, conv3_b = params.conv3
  x = jax.nn.relu(
    (
      jax.lax.conv_general_dilated(
        lhs=x,
        rhs=conv3_w,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NHWC", "OIHW", "NHWC"),
      )
      + jnp.reshape(conv3_b, (1, 1, 1, -1))
    )
  )

  # fourth convolution: [BS x 8 x 8 x 64] -> [BS x 6 x 6 x 64]
  conv4_w, conv4_b = params.conv4
  x = jax.nn.relu(
    jax.lax.conv_general_dilated(
      lhs=x,
      rhs=conv4_w,
      window_strides=(1, 1),
      padding="VALID",
      dimension_numbers=("NHWC", "OIHW", "NHWC"),
    )
    + jnp.reshape(conv4_b, (1, 1, 1, -1))
  )

  # second batch normalization: [BS x 6 x 6 x 64]
  gamma2, beta2 = params.bn2
  if is_training:
    # compute batch stats and update running stats
    batch_mean2 = jnp.mean(x, axis=(0, 1, 2))
    batch_var2 = jnp.var(x, axis=(0, 1, 2))

    # normalize using batch stats
    x = (x - jnp.reshape(batch_mean2, (1, 1, 1, -1))) / jnp.sqrt(
      jnp.reshape(batch_var2, (1, 1, 1, -1)) + 1e-5
    )

    # update running stats with momentum
    new_mean2 = momentum * bn_state.mean2 + (1 - momentum) * batch_mean2
    new_var2 = momentum * bn_state.var2 + (1 - momentum) * batch_var2

  else:  # use stored running stats for inference
    x = (x - jnp.reshape(bn_state.mean2, (1, 1, 1, -1))) / jnp.sqrt(
      jnp.reshape(bn_state.var2, (1, 1, 1, -1)) + 1e-5
    )
  # apply scale and shift
  x = jnp.reshape(gamma2, (1, 1, 1, -1)) * x + jnp.reshape(beta2, (1, 1, 1, -1))

  # maxpool: [BS x 6 x 6 x 64] -> [BS x 3 x 3 x 64]
  x = jax.lax.reduce_window(
    operand=x,
    init_value=-jnp.inf,
    computation=jax.lax.max,
    window_dimensions=(1, 2, 2, 1),
    window_strides=(1, 2, 2, 1),
    padding="VALID",
  )

  # flatten: [BS x 3 x 3 x 64] -> [BS x 576]
  x = x.reshape(x.shape[0], -1)

  # fully connected: [BS x 576] -> [BS x 10]
  fc_w, fc_b = params.fc
  logits = jnp.dot(x, fc_w.T) + fc_b

  if is_training:
    new_bn_state = BatchNormState(
      mean1=new_mean1,
      var1=new_var1,
      mean2=new_mean2,
      var2=new_var2,
    )
    return logits, new_bn_state

  return logits, bn_state


@jit
def ce_loss(logits: jax.Array, y: jax.Array) -> jax.Array:
  """
  Compute cross-entropy loss.

  Args:
    logits: Outputted logits of the forward pass.
    y: Labels associated with the logit data.

  Returns:
    A JAX Array containing the loss value.
  """
  y_onehot = jax.nn.one_hot(y, num_classes=10)
  log_probs = jax.nn.log_softmax(logits)
  return -jnp.mean(jnp.sum(log_probs * y_onehot, axis=-1))


@jit
def train_step(
  params: ModelParams,
  bn_state: BatchNormState,
  x: jax.Array,
  y: jax.Array,
  lr: float,
) -> Tuple[ModelParams, BatchNormState, jax.Array]:
  """
  Perform a single training step.

  Args:
    params: Dictionary of model parameters.
    bn_state: Batch normalization state.
    x: Input data [BS x 28 x 28 x 1] in NHWC format or [BS x 784] flattened.
    y: Labels for the input data.
    lr: Learning rate.

  Returns:
    A Tuple of (updated model parameters, updated batchnorm state, loss value)
  """

  # custom loss function that computes gradients with updated batchnorm state
  def loss_with_state(p: ModelParams) -> Tuple[jax.Array, BatchNormState]:
    logits, new_bn_state = forward(p, bn_state, x, is_training=True)
    loss_val = ce_loss(logits, y)
    return loss_val, new_bn_state

  # compute gradients and get loss value and updated state in one pass
  (loss_val, new_bn_state), grads = jax.value_and_grad(loss_with_state, has_aux=True)(
    params
  )

  # apply gradients to update params
  new_params = ModelParams(
    conv1=tuple(w - lr * dw for w, dw in zip(params.conv1, grads.conv1)),
    conv2=tuple(w - lr * dw for w, dw in zip(params.conv2, grads.conv2)),
    conv3=tuple(w - lr * dw for w, dw in zip(params.conv3, grads.conv3)),
    conv4=tuple(w - lr * dw for w, dw in zip(params.conv4, grads.conv4)),
    bn1=tuple(w - lr * dw for w, dw in zip(params.bn1, grads.bn1)),
    bn2=tuple(w - lr * dw for w, dw in zip(params.bn2, grads.bn2)),
    fc=tuple(w - lr * dw for w, dw in zip(params.fc, grads.fc)),
  )

  return new_params, new_bn_state, loss_val


@jit
def accuracy(
  params: ModelParams, bn_state: BatchNormState, x: jax.Array, y: jax.Array
) -> jax.Array:
  """
  Compute the model's accuracy.

  Args:
    params: Model parameters.
    bn_state: Batch normalization state.
    x: Input data [BS x 28 x 28 x 1] in NHWC format or [BS x 784] flattened.
    y: Labels for the input data.

  Returns:
    A JAX Array containing a scalar value of the average accuracy across a batch.
  """
  logits, _ = forward(params, bn_state, x, is_training=False)
  preds = jnp.argmax(logits, axis=-1)
  return jnp.mean(preds == y)


# program entry point
if __name__ == "__main__":
  # load dataset
  X_train, Y_train, X_test, Y_test = load_mnist("./mnist")

  # hyperparameters
  learning_rate = 1e-2
  num_epochs = 50
  batch_size = 64

  # initialize model parameters
  params, bn_state = init_model()

  # number of batches
  num_train = X_train.shape[0]
  num_batches = num_train // batch_size

  # training loop
  print("Starting CNN training...")
  for epoch in range(num_epochs):
    # shuffle training data
    key = jax.random.PRNGKey(epoch)  # different key for each epoch
    perm = jax.random.permutation(key, num_train)
    X_train_shuffled = X_train[perm]
    Y_train_shuffled = Y_train[perm]

    # mini-batch training
    epoch_loss = 0.0
    for batch in tqdm.trange(num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}"):
      start_idx = batch * batch_size
      end_idx = start_idx + batch_size
      X_batch = X_train_shuffled[start_idx:end_idx]
      Y_batch = Y_train_shuffled[start_idx:end_idx]

      # training step: update parameters and computer batch loss
      params, bn_state, batch_loss = train_step(
        params, bn_state, X_batch, Y_batch, learning_rate
      )
      epoch_loss += batch_loss / num_batches

    # end-of-epoch valuation
    train_accuracy = accuracy(params, bn_state, X_train, Y_train)
    test_accuracy = accuracy(params, bn_state, X_test, Y_test)

    print(
      f"Train Loss: {epoch_loss:.4f}, "
      f"Train Accuracy: {train_accuracy * 100:.2f}%, "
      f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
    )

  # final test set evaluation
  final_test_accuracy = accuracy(params, bn_state, X_test, Y_test)
  print("\nTraining complete!")
  print(f"Final Test Accuracy: {final_test_accuracy * 100:.2f}%")
