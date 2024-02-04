"""Module for Early Stopping callback"""

import os

import numpy as np
import torch
import torch.nn as nn

from callbacks.callback import Callback


class EarlyStopping(Callback):
    """
    Callback to stop training when a monitored metric has stopped improving.
    
    Attributes:
        patience (int): Number of epochs to wait after last time the monitored metric improved.
        metric (str): Name of the metric to be monitored.
        metric_ascending (bool): True if metric improvement is an increase, False if decrease.
        best_metric_val (float): Best value of the monitored metric.
        early_stop (bool): Flag to signal that early stopping condition was met.
    
    Parameters:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        metric (str): The metric name to be monitored.
        metric_ascending (bool, optional): Specifies if the monitored metric is expected to
            increase (True) or decrease (False). Defaults to True.
    """
    def __init__(self, patience: int, metric: str, metric_ascending=True):
        self.patience = patience
        self.metric = metric
        self.counter = 0
        self.metric_ascending = metric_ascending
        self.best_metric_val = -np.inf if metric_ascending else np.inf
        self.early_stop = False

    def on_epoch_end(self, epoch: int, logs=None):
        """
        Check if the training should be stopped early at the end of each epoch.
        
        Parameters:
            epoch (int): The current epoch number.
            logs (dict): A dictionary containing the training and validation metrics.
        """
        metric_val = logs.get(self.metric)
        if (self.metric_ascending and metric_val > self.best_metric_val) or (
            not self.metric_ascending and metric_val < self.best_metric_val
        ):
            self.best_metric_val = metric_val
            self.counter = 0
            self.save_best_model(logs.get("model"))
        else:
            self.counter += 1

        if self.patience == self.counter:
            self.early_stop = True

    def save_best_model(self, model: nn.Module):
        """
        Save the model when it achieves a new best value for the monitored metric.
        
        Parameters:
            model (nn.Module): The model to save.
        """
        if model is None:
            return
        result_directory = os.path.join("results", "best")
        path_to_model_file = os.path.join(result_directory, "best.pt")
        if os.path.exists(result_directory):
            torch.save(model.state_dict(), path_to_model_file)
        else:
            os.makedirs(result_directory)
            torch.save(model.state_dict(), path_to_model_file)
