from typing import Tuple, Dict
import numpy as np
import tensorflow as tf


class VAEDataModule:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def to_dataset(self, inputs: Dict[str, np.ndarray]) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices(inputs)
        return ds.batch(self.batch_size)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.3,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    from sklearn.model_selection import train_test_split

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(X, y, test_size=test_size)

    X_train = _flatten(X=X_train_raw)
    X_val = _flatten(X=X_val_raw)

    train_inputs = _build_inputs(X_train, y_train)
    val_inputs = _build_inputs(X_val, y_val)

    return train_inputs, val_inputs


def _flatten(X: np.ndarray) -> np.ndarray:
    nt, ny, nx = X.shape
    return np.reshape(X, (nt, ny * nx), order="F")


def _build_inputs(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, np.ndarray]:
    dummy = np.ones((X.shape[0], 1))

    return {
        "x": X,
        "dummy": dummy,
        "r": y,
    }
