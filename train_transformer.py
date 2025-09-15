import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import torchmetrics
from torch.utils.data import DataLoader

from adopt import ADOPT
from data.jointdata import JointData
from models.marker_transformer import MarkerClassification
from trainer import Trainer
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
        "--batch_size", type=int, default=8, help="Batch size for data loading/training"
    )
    parser.add_argument(
        "--interval_size",
        type=int,
        default=64,
        help="Interval size for data processing",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for the optimizer"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Timeout in seconds for training"
    )
    parser.add_argument(
        "--pfe_dim", type=int, default=128, help="Pose feature extractor dimension"
    )
    parser.add_argument(
        "--fc_dim", type=int, default=128, help="Neural feature classifier dimension"
    )
    parser.add_argument(
        "--pfe_head", type=int, default=2, help="Number of heads of pfe"
    )
    parser.add_argument("--fc_head", type=int, default=2, help="Number of heads of nfc")
    parser.add_argument(
        "--save_folder", type=str, default="./out", help="Save folder for outputs"
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
    parser.add_argument("--cross", type=int, default=0, help="Cross-validation index")

    args = parser.parse_args()
    batch_size: int = args.batch_size
    interval_size: int = args.interval_size
    learning_rate: float = args.lr
    num_epochs: int = args.epochs
    timeout: int = args.timeout
    pfe_dim: int = args.pfe_dim
    fc_dim: int = args.fc_dim
    pfe_head: int = args.pfe_head
    fc_head: int = args.fc_head
    save_folder: str = args.save_folder
    cross_index: int = args.cross
    seed: int = args.seed
    pos_ratio: bool = args.use_pos_ratio
    base_folder: str = args.base_folder
    on_cluster: bool = args.cluster

    # Create a string representation of the layer configurations
    if save_folder is not None:
        save_folder = save_folder.rstrip("/") + "/"
    if base_folder is not None:
        base_folder = base_folder.rstrip("/") + "/"
        os.makedirs(base_folder, exist_ok=True)

    # Initialise the logger
    logger = Logger(
        f"{interval_size}_{pfe_dim}_{pfe_head}_{fc_dim}_{fc_head}", cross_index
    )

    # Seed randomness for reproducibility
    g = seed_randomness(seed)

    # Load or initialize the training dataset
    train_ds_file = f"{base_folder}trans_train_{cross_index}.pt"
    if os.path.isfile(train_ds_file):
        train_ds = torch.load(train_ds_file, weights_only=False)
    else:
        train_ds = JointData(
            sequence_interval=interval_size,
            permute_markers=True,
            base_directory=base_folder,
            cross_validation_index=cross_index,
        )
        torch.save(train_ds, train_ds_file)

    # Load or initialize the evaluation dataset
    eval_ds_file = f"{base_folder}trans_eval_{cross_index}.pt"
    if os.path.isfile(eval_ds_file):
        eval_ds = torch.load(eval_ds_file, weights_only=False)
    else:
        eval_ds = JointData(
            sequence_interval=interval_size,
            permute_markers=True,
            is_trainings_dataset=False,
            base_directory=base_folder,
            cross_validation_index=cross_index,
        )
        torch.save(eval_ds, eval_ds_file)

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
    model = MarkerClassification(
        output_dim=1,
        feature_classifier_dim=fc_dim,
        pose_feature_dim=pfe_dim,
        num_classifier_heads=fc_head,
        num_feature_heads=pfe_head,
        interval_length=interval_size,
    ).to(DEVICE)

    # Get class ratio for weighted loss
    ratio = train_ds.get_healthy_sick_ratio()

    # Initialize loss function
    loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=ratio if pos_ratio else None).to(
        DEVICE
    )

    # Initialize optimizer and trainer
    optimizer = ADOPT(model.parameters(), lr=learning_rate)
    trainer = Trainer(
        model=model, optimizer=optimizer, num_epochs=num_epochs, timeout=timeout
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
        run_type="standard",
        save_folder=save_folder,
        progress_bar=not on_cluster,
    )

    # Save the final model checkpoint
    date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    torch.save(
        model,
        f"{save_folder}transformer_{date_time}.ckpt",
    )

    # Log results and save metrics
    logger.log(
        eval_loss,
        eval_metrics,
        train_loss,
        train_metrics,
        seed=seed,
        positive_ratio_used=pos_ratio,
        pfe_dimension=pfe_dim,
        fc_dimension=fc_dim,
        learning_rate=learning_rate,
        feat_head=pfe_head,
        fc_head=fc_head,
        interval_length=interval_size,
        crossid=cross_index,
        batch_size=batch_size,
    )

    logger.save_metrics(save_folder)
    logger.save_tensorboard(f"{save_folder}tensorboard_transformer")

    # Save the traced model if not on a cluster
    if not on_cluster:
        traced_model = torch.jit.script(model)
        traced_model.save(
            f"{save_folder}{interval_size}_{feat_dim}_{feat_head}_{merg_dim}_{merg_head}_{date_time}.pt"
        )


if __name__ == "__main__":
    main()
