import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def format_input_to_dataset(
    inputs: dict[str, np.ndarray], batch_size: int
) -> tf.data.Dataset:
    """
    Convert a dictionary of NumPy arrays into a batched TensorFlow dataset,
    for which each value must contain the same number of samples along
    the first dimension.

    :param inputs:      A dictionary mapping feature names to NumPy arrays.
    :param batch_size:  The number of samples per batch.
    :return:            A batched ``tf.data.Dataset`` yielding dictionaries of tensors.
    """
    ds = tf.data.Dataset.from_tensor_slices(inputs)
    return ds.batch(batch_size)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Split the data into training and validation sets.

    :param X:         The input data to split.
    :param y:         The target data to split.
    :param test_size: The proportion of the data to be used for validation.
    :return:          A tuple containing the training and validation sets.
    """
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=test_size)

    X_train = flatten_input(X=X_train_raw)
    X_val = flatten_input(X=X_val_raw)

    train_inputs = _build_inputs(X_train, y_train)
    val_inputs = _build_inputs(X_val, y_val)

    return train_inputs, val_inputs


def flatten_input(X: np.ndarray) -> np.ndarray:
    """
    Flatten the two last dimensions of a 3-dimensional input array into a single dimension.

    :param X:   The 3-dimensional input array to flatten.
    :return:    The flattened, 2-dimensional array.
    """
    nt, ny, nx = X.shape
    return np.reshape(X, (nt, ny * nx))


def unflatten_input(X: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """
    Reshape a 2-dimensional input array into a 3-dimensional array.

    :param X:   The 2-dimensional input array to reshape.
    :param ny:  The size of the second dimension in the output array (number of latitudes).
    :param nx:  The size of the third dimension in the output array (number of longitudes).
    :return:    The reshaped, 3-dimensional array.
    """
    nt, nxy = X.shape
    if nxy != ny * nx:
        raise ValueError(
            f"Input array has {nxy} features, but expected {ny * nx} features "
            f"for the specified dimensions (ny={ny}, nx={nx})."
        )
    return np.reshape(X, (nt, ny, nx))


def _build_inputs(
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    """Format the input arrays as a dictionary to feed to the model."""
    dummy = np.ones((X.shape[0], 1))
    return {
        "x": X,
        "dummy": dummy,
        "target": y,
    }
