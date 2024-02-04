"""
Module: Callbacks for Training Process

This module introduces a `Callback` class to inject custom behavior into various stages of a 
machine learning training cycle. The class provides hooks at the start/end of training and at the
beginning/end of each epoch, allowing for custom actions like logging, checkpointing, or adjusting
learning rates without altering the main training loop.
"""

class Callback:
    """
    Base class for creating callbacks to monitor and act on different stages of the training 
    process.

    Callbacks offer a way to execute custom code at key points in training, such as the start/end of
    the entire training process, and the beginning/end of each epoch. This functionality is useful 
    for tasks like logging metrics, saving model checkpoints, early stopping, or dynamically 
    adjusting learning parameters.

    Methods to override:
    - on_train_begin(logs=None): Executed once before training starts.
    - on_train_end(logs=None): Executed once after training ends.
    - on_epoch_begin(epoch, logs=None): Called at the start of every epoch.
    - on_epoch_end(epoch, logs=None): Called at the end of every epoch.

    Parameters:
    - epoch: Integer, the index of the current epoch.
    - logs: Dict, contains training and validation metrics.

    Extend this class to implement custom behavior at specified training stages.
    """
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""

    def on_train_end(self, logs=None):
        """Called at the end of training."""

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of an epoch."""

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
