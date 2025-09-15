import torch
import os
import datetime
import torchmetrics

from torch.utils.data import DataLoader

from models.mlp import MultiMLP, LRPMLP
import argparse
from data.jointdata import JointData

from trainer import Trainer
from adopt import ADOPT
import json
import numpy as np
import random
from itertools import chain

from logger import Logger

from util import seed_randomness

# Use GPU if it's available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """
    Main function to execute the training and evaluation of the model.
    Sets up the data, model, optimizer, and metrics, and then starts the training process.
    """
    parser = argparse.ArgumentParser(description="Process some parameters")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for data loading/training",
    )
    parser.add_argument(
        "--interval_size",
        type=int,
        default=128,
        help="Interval size for data processing",
    )
    parser.add_argument(
        "--epochs", type=int, default=75, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds for training"
    )
    parser.add_argument(
        "--pfe_layers",
        nargs="+",
        type=int,
        default=[256, 512, 256, 64],
        help="Pose feature extractor layer sizes",
    )
    parser.add_argument(
        "--fc_layers",
        nargs="+",
        type=int,
        default=[256, 512, 256, 64],
        help="Neural feature classifier layer sizes",
    )
    parser.add_argument(
        "--use_pos_ratio",
        action="store_true",
        help="Whether to use positive ratio in loss function",
    )
    parser.add_argument(
        "--cluster", action="store_true", help="Whether code runs on the cluster"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for random number generators"
    )
    parser.add_argument(
        "--base_folder",
        type=str,
        default="./",
        help="Base folder path for data storage",
    )
    parser.add_argument(
        "--save_folder", type=str, default="./out", help="Save folder for outputs"
    )
    parser.add_argument("--cross", type=int, default=0, help="Cross-validation index")
    parser.add_argument(
        "--lrp", action="store_true", help="Training the lrp version"
    )  # The internal structure differs due to the LRP framework
    args = parser.parse_args()

    # Extract and process command line arguments
    batch_size: int = args.batch_size
    interval_size: int = args.interval_size
    learning_rate: float = args.lr
    num_epochs: int = args.epochs
    pfe_layers: list[int] = args.pfe_layers
    fc_layers: list[int] = args.fc_layers
    timeout: int = args.timeout
    seed: int = args.seed
    pos_ratio: bool = args.use_pos_ratio
    base_folder: str = args.base_folder
    save_folder: str = args.save_folder
    cross_index: int = args.cross
    on_cluster: bool = args.cluster
    train_lrp: bool = args.lrp

    # Ensure save and base folders end with a slash and exist
    if save_folder is not None:
        save_folder = save_folder.rstrip("/") + "/"
    if base_folder is not None:
        base_folder = base_folder.rstrip("/") + "/"
        os.makedirs(base_folder, exist_ok=True)

    # Seed randomness for reproducibility
    g = seed_randomness(seed)

    # Create a string representation of the layer configurations
    layerstring = ""
    for l in pfe_layers:
        layerstring += str(l) + "_"
    for l in fc_layers:
        layerstring += str(l) + "_"
    layerstring = layerstring[:-1]  # Remove trailing underscore

    # Initialize the logger
    logger = Logger(f"{layerstring}", cross_index)

    # Load or initialize the training dataset
    train_file = f"{base_folder}mlp_train_{cross_index}.pt"
    if os.path.isfile(train_file):
        train_ds = torch.load(train_file, weights_only=False)
    else:
        train_ds = JointData(
            sequence_interval=interval_size,
            permute_markers=False,
            base_directory=base_folder,
            cross_validation_index=cross_index,
        )
        torch.save(train_ds, train_file)

    # Load or initialize the evaluation dataset
    eval_file = f"{base_folder}mlp_eval_{cross_index}.pt"
    if os.path.isfile(eval_file):
        eval_ds = torch.load(eval_file, weights_only=False)
    else:
        eval_ds = JointData(
            sequence_interval=interval_size,
            is_trainings_dataset=False,
            base_directory=base_folder,
            cross_validation_index=cross_index,
        )
        torch.save(eval_ds, eval_file)

    # Setup data loaders
    train_loader = DataLoader(
        train_ds,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size,
        generator=g,
    )
    val_loader = DataLoader(
        eval_ds,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size,
        generator=g,
    )

    # Initialize the model
    if train_lrp:
        model = LRPMLP(
            interval_length=interval_size,
            pfe_dims=pfe_layers,
            fc_dims=fc_layers,
            num_marker=train_ds.num_markers,
        ).to(DEVICE)
    else:
        model = MultiMLP(
            interval_length=interval_size,
            pfe_dims=pfe_layers,
            fc_dims=fc_layers,
            num_marker=train_ds.num_markers,
        ).to(DEVICE)
    # Get class ratio for weighted loss
    ratio = train_ds.get_healthy_sick_ratio()
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=ratio if pos_ratio else None).to(
        DEVICE
    )

    # Initialize optimizer and trainer
    optimizer = ADOPT(model.parameters(), lr=learning_rate)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        num_epochs=num_epochs,
        timeout=timeout,
    )

    # Setup training and evaluation metrics
    training_metrics = [
        torchmetrics.Accuracy(task="binary").to(DEVICE),
        torchmetrics.Precision(task="binary").to(DEVICE),
        torchmetrics.Recall(task="binary").to(DEVICE),
        torchmetrics.Specificity(task="binary").to(DEVICE),
        torchmetrics.F1Score(task="binary").to(DEVICE),
    ]
    evaluation_metrics = (
        torchmetrics.Accuracy(task="binary").to(DEVICE),
        torchmetrics.Precision(task="binary").to(DEVICE),
        torchmetrics.Recall(task="binary").to(DEVICE),
        torchmetrics.Specificity(task="binary").to(DEVICE),
        torchmetrics.F1Score(task="binary").to(DEVICE),
    )  # Using the same metrics for evaluation

    # Train the model
    train_loss, eval_loss, train_metrics, eval_metrics, checkpoints = trainer.train(
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        train_metrics=training_metrics,
        eval_metrics=evaluation_metrics,
        loss_func=loss_func,
        run_type="lrp" if train_lrp else "standard",
        save_folder=save_folder,
        progress_bar=not on_cluster,
    )

    # Save the final model checkpoint
    date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    torch.save(
        model,
        f"{save_folder}/mlp_{date_time}.ckpt",
    )

    # Log results and save metrics
    logger.log(
        eval_loss,
        eval_metrics,
        train_loss,
        train_metrics,
        seed=seed,
        positive_ratio_used=pos_ratio,
        fc_layers=fc_layers,
        pfe_layers=pfe_layers,
        learning_rate=learning_rate,
        interval_length=interval_size,
        crossid=cross_index,
        batch_size=batch_size,
    )

    logger.save_metrics(save_folder)
    logger.save_tensorboard(f"{save_folder}tensorboard_transformer")


if __name__ == "__main__":
    main()
