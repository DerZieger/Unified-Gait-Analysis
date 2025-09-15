import os
from torch.utils.tensorboard import SummaryWriter
import torch


class Logger:
    """
    A logger class for saving training and evaluation metrics,
    and managing TensorBoard logs for machine learning experiments.
    """

    def __init__(self, name: str, cross_index: int):
        """
        Initializes the Logger.

        Parameters:
        - name (str): The name or identifier for the logging run.
        - cross_index (int): Index for cross-validation runs.
        """
        self.name = name
        self.cross_index = cross_index
        self.logging_dict = {}
        self.data_logged = False

    def log(
        self,
        eval_loss: list,
        eval_metrics: list,
        train_loss: list,
        train_metrics: list,
        **kwargs,
    ):
        """
        Logs the given metrics for each epoch.

        Parameters:
        - eval_loss (list): List of evaluation losses per epoch.
        - eval_metrics (list): List of evaluation metrics (accuracy, precision, recall, specificity, f1).
        - train_loss (list): List of training losses per epoch.
        - train_metrics (list): List of training metrics (accuracy, precision, recall, specificity, f1).
        - kwargs: Additional key-value pairs to log.
        """
        for keyword in kwargs:
            self.logging_dict[keyword] = kwargs[keyword]
        self.logging_dict["epochs"] = len(eval_loss)
        for epoch in range(len(eval_loss)):
            self.logging_dict[str(epoch)] = {
                "eval": {
                    "loss": eval_loss[epoch],
                    "accuracy": eval_metrics[epoch][0].item(),
                    "precision": eval_metrics[epoch][1].item(),
                    "recall": eval_metrics[epoch][2].item(),
                    "specificity": eval_metrics[epoch][3].item(),
                    "f1": eval_metrics[epoch][4].item(),
                },
                "train": {
                    "loss": train_loss[epoch],
                    "accuracy": train_metrics[epoch][0].item(),
                    "precision": train_metrics[epoch][1].item(),
                    "recall": train_metrics[epoch][2].item(),
                    "specificity": train_metrics[epoch][3].item(),
                    "f1": train_metrics[epoch][4].item(),
                },
            }
        self.data_logged = True

    def save_metrics(self, save_folder: str = "./out"):
        """
        Saves the logged metrics to a JSON file.

        Parameters:
        - save_folder (str): Directory to save the metrics.
        """
        if not self.data_logged:
            print("Log data before saving metrics")
            return

        import json
        import datetime

        if save_folder:
            save_folder = save_folder.rstrip("/") + "/"
        os.makedirs(save_folder, exist_ok=True)

        date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with open(f"{save_folder}metrics_{self.name}_{date_time}.json", "w") as outfile:
            json.dump(self.logging_dict, outfile, indent=4, sort_keys=True)

    def initialize_tensorboard(self, tensorboard_folder: str = "./tensorboard"):
        """
        Initializes TensorBoard for logging.

        Parameters:
        - tensorboard_folder (str): Directory to save TensorBoard logs.
        """
        if tensorboard_folder:
            tensorboard_folder = tensorboard_folder.rstrip("/") + "/"
        os.makedirs(tensorboard_folder, exist_ok=True)
        self.writer = SummaryWriter(
            f"{tensorboard_folder}{self.name}_{self.cross_index}"
        )

    def save_network_graph(
        self,
        model: torch.nn.Module,
        model_input: torch.Tensor,
        tensorboard_folder: str = "./tensorboard",
    ):
        """
        Saves the network graph to TensorBoard.

        Parameters:
        - model (torch.nn.Module): The network model.
        - model_input (torch.Tensor): Input tensor used to trace the model.
        - tensorboard_folder (str): Directory to save TensorBoard logs.
        """
        if not hasattr(self, "writer"):
            self.initialize_tensorboard(tensorboard_folder)
        self.writer.add_graph(model, model_input)

    def save_tensorboard(self, tensorboard_folder: str = "./tensorboard"):
        """
        Saves scalar values to TensorBoard.

        Parameters:
        - tensorboard_folder (str): Directory to save TensorBoard logs.
        """
        if not self.data_logged:
            print("Log data before saving tensorboard")
            return

        if not hasattr(self, "writer"):
            self.initialize_tensorboard(tensorboard_folder)

        for epoch in range(self.logging_dict["epochs"]):
            epoch_str = str(epoch)

            for phase in ["train", "eval"]:
                for metric in [
                    "loss",
                    "f1",
                    "accuracy",
                    "precision",
                    "recall",
                    "specificity",
                ]:
                    self.writer.add_scalar(
                        f"{phase.capitalize()} {metric.capitalize()}/Layer/{self.name}",
                        self.logging_dict[epoch_str][phase][metric],
                        epoch,
                    )
                    self.writer.add_scalar(
                        f"{phase.capitalize()} {metric.capitalize()}/Cross/{self.cross_index}",
                        self.logging_dict[epoch_str][phase][metric],
                        epoch,
                    )

        self.writer.close()
