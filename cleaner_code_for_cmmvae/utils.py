import matplotlib.pyplot as plt
import numpy as np


def plot_losses(training_loss: np.ndarray, validation_loss: np.ndarray):
    """
    Plot the training loss and validation loss.

    :param training_loss:   The training loss.
    :param validation_loss: The validation loss.
    """
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    plt.title(label="Model Loss by Epoch", loc="center")
    ax.plot(training_loss, label="Training Data", color="white")
    ax.plot(validation_loss, label="Test Data", color="red")
    ax.set(xlabel="Epoch", ylabel="Loss")
    plt.legend()
    plt.show()
