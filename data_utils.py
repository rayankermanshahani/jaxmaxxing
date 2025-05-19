import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple


def load(file_path: str) -> np.ndarray:
  """
  Read raw MNIST data from ubyte files into NumPy arrays.
  """
  abs_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
  try:
    with open(abs_file_path, "rb") as f:
      data = np.frombuffer(f.read(), dtype=np.uint8)
  except FileNotFoundError:
    print(f"Error finding file: {file_path}")
    raise
  except IOError:
    print(f"Error reading file: {file_path}")
    raise
  return data


def load_mnist(data_dir: str) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """
  Returns a tuple of JAX Arrays containing the training samples and labels and the test samples and labels.
    X_train, Y_train, X_test, Y_test
  """
  # load data into numpy arrays
  X_train = load(os.path.join(data_dir, "train-images-idx3-ubyte"))[0x10:].reshape(
    (-1, 28 * 28)
  )
  Y_train = load(os.path.join(data_dir, "train-labels-idx1-ubyte"))[8:]
  X_test = load(os.path.join(data_dir, "t10k-images-idx3-ubyte"))[0x10:].reshape(
    (-1, 28 * 28)
  )
  Y_test = load(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))[8:]

  # load numpy arrays into jax arrays
  X_train = jnp.array(X_train, dtype=jnp.float32) / 255.0  # normalize images
  Y_train = jnp.array(Y_train, dtype=jnp.int32)
  X_test = jnp.array(X_test, dtype=jnp.float32) / 255.0  # normalize images
  Y_test = jnp.array(Y_test, dtype=jnp.int32)

  return X_train, Y_train, X_test, Y_test
