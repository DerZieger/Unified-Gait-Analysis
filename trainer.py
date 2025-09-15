from typing import List
from tqdm.autonotebook import tqdm
import os
import sys
import time


class Trainer:
    def __init__(self, model, optimizer, num_epochs: int = 100, timeout: int = None):
        """
        Initializes the Trainer class.

        Parameters:
        model: The neural network model to be trained.
        optimizer: The optimization algorithm for updating model parameters.
        num_epochs (int): The number of epochs for training. Default is 100.
        timeout (int, optional): The maximum duration for training in seconds. Default is None.
        """
        self.model = model
        self.optimizer = optimizer
        self.epochs = num_epochs
        self.timeout = timeout
        self.device = next(model.parameters()).device
        self.runfunc = {
            "standard": self.run_with_three_parameter,
            "lrp": self.run_with_one_parameter,
        }
        self.num_model_params = 0
        for param in model.parameters():
            self.num_model_params += param.flatten().shape[0]

    def train(
        self,
        train_dataloader,
        eval_dataloader,
        loss_func,
        train_metrics: List = [],
        eval_metrics: List = [],
        save_folder: str = "./ckpts/",
        save_name: str = None,
        pipe=sys.stderr,
        progress_bar: bool = True,
        run_type: str = "standard",
    ):
        """
        Trains the model using the provided data and parameters.

        Parameters:
        train_dataloader: DataLoader for training data.
        eval_dataloader: DataLoader for evaluation data.
        loss_func: Loss function to be used for training and evaluation.
        train_metrics (List): List of metrics to compute during training.
        eval_metrics (List): List of metrics to compute during evaluation.
        save_folder (str): Folder to save model checkpoints. Default is "./transformer_ckpts/".
        save_name (str, optional): Filename prefix for saved checkpoints. Default is None.
        pipe: Output stream for printing progress. Default is sys.stderr.
        progress_bar (bool): Whether to display a progress bar. Default is True.
        run_type (str): Selection of the run method (0, 1, or 2). Default is 0.

        Returns:
        Tuple containing training losses, evaluation losses, training metrics, evaluation metrics,
        and state dicts.
        """
        start_time = time.perf_counter()
        eval_metric_list = []
        train_metric_list = []
        eval_losses = []
        train_losses = []
        state_dicts = []
        os.makedirs(save_folder, exist_ok=True)

        if not progress_bar:
            pipe = f = open(os.devnull, "w")
        print(
            "-This Model Has %d (Approximately %d Million) Parameters!"
            % (self.num_model_params, self.num_model_params // 1e6),
            file=pipe,
        )

        for epoch in (pbar := tqdm(range(self.epochs), file=pipe)):
            train_loss, t_metrics = self.runfunc[run_type](
                train_dataloader, loss_func, train_metrics, pipe, train=True
            )
            train_losses.append(train_loss)
            train_metric_list.append(t_metrics)

            eval_loss, e_metrics = self.runfunc[run_type](
                eval_dataloader, loss_func, eval_metrics, pipe, train=False
            )
            eval_losses.append(eval_loss)
            eval_metric_list.append(e_metrics)

            pbar.set_description_str(
                f"Train_loss: {train_loss:.4f}, Eval_loss: {eval_loss:.4f}"
            )

            state_dicts.append(self.model.state_dict())

            if save_name is not None:
                import torch

                torch.save(
                    state_dicts[epoch],
                    save_folder + save_name + f"_{epoch:05d}.ckpt",
                )
            if self.timeout is not None and self.timeout < (
                time.perf_counter() - start_time
            ):
                break

        return (
            train_losses,
            eval_losses,
            train_metric_list,
            eval_metric_list,
            state_dicts,
        )

    def run_with_three_parameter(
        self, data_loader, loss_func, metrics, pipe, train: bool = True
    ):
        """
        Executes training or evaluation for Timeseries data with marker IDs and masks.

        Parameters:
        data_loader: DataLoader with input data and ground truth labels.
        loss_func: Loss function to be used.
        metrics: List of metrics to compute.
        pipe: Output stream for progress.
        train (bool): Flag indicating if training (True) or evaluation (False) is performed.

        Returns:
        Tuple containing the running average loss and computed metrics.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()
        running_average = 0.0
        count = 0

        for data, gt_label in (pbar := tqdm(data_loader, leave=False, file=pipe)):
            if train:
                self.optimizer.zero_grad()
            time_series, mark_ids, masks = data
            time_series = time_series.to(self.device)
            mark_ids = mark_ids.to(self.device)
            masks = masks.to(self.device)
            gt_label = gt_label.to(self.device)

            pred_label = self.model(time_series, mark_ids, masks)
            pred_label = pred_label.squeeze(1)
            loss = loss_func(pred_label, gt_label.float())

            if train:
                loss.backward()
                self.optimizer.step()

            for met in metrics:
                met(pred_label, gt_label)
            running_average += loss.item()
            count += time_series.shape[0]

            pbar.set_description(f"{loss.item():.3f}")

        return_metric = [metric.compute() for metric in metrics]
        return running_average / count, return_metric

    def run_with_one_parameter(
        self, data_loader, loss_func, metrics, pipe, train: bool = True
    ):
        """
        Executes training or evaluation for Timeseries data for lrp version of mlp.

        Parameters:
        data_loader: DataLoader with input data and ground truth labels.
        loss_func: Loss function to be used.
        metrics: List of metrics to compute.
        pipe: Output stream for progress.
        train (bool): Flag indicating if training (True) or evaluation (False) is performed.

        Returns:
        Tuple containing the running average loss and computed metrics.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()
        running_average = 0.0
        count = 0

        for data, gt_label in (pbar := tqdm(data_loader, leave=False, file=pipe)):
            if train:
                self.optimizer.zero_grad()
            time_series, _, __ = data
            time_series = time_series.to(self.device)
            gt_label = gt_label.to(self.device)

            pred_label = self.model(time_series)
            pred_label = pred_label.squeeze(1)
            loss = loss_func(pred_label, gt_label.float())

            if train:
                loss.backward()
                self.optimizer.step()

            for met in metrics:
                met(pred_label, gt_label)
            running_average += loss.item()
            count += time_series.shape[0]

            pbar.set_description(f"{loss.item():.3f}")

        return_metric = [metric.compute() for metric in metrics]
        return running_average / count, return_metric
