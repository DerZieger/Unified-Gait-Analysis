import json
import os
import math

import numpy as np
import torch
from torch.utils.data import Dataset

from supr.pytorch.supr import SUPR

from util import load_data

"""
                       ┌──┐                      
                       │15│                      
                       └┬─┘                      
┌──┐  ┌──┐  ┌──┐  ┌──┐ ┌┴─┐┌──┐  ┌──┐  ┌──┐  ┌──┐                    
│20├──┤18├──┤16├──┤13│ │12││14├──┤17├──┤19├──┤21│                    
└──┘  └──┘  └──┘  └─┬┘ └┬─┘└┬─┘  └──┘  └──┘  └──┘                      
                    │  ┌┴┐  │                     
                    └──┤9├──┘                       
                       └┬┘                       
                       ┌┴┐                       
                       │6│                       
                       └┬┘                       
                       ┌┴┐                       
                       │3│                       
                       └┬┘                       
                       ┌┴┐                       
                    ┌──┤0├──┐                    
                    │  └─┘  │                    
                   ┌┴┐     ┌┴┐                   
                   │1│     │2│                   
                   └┬┘     └┬┘                   
                   ┌┴┐     ┌┴┐                   
                   │4│     │5│                   
                   └┬┘     └┬┘                   
                   ┌┴┐     ┌┴┐                   
                   │7│     │8│                   
                   └┬┘     └┬┘                   
                  ┌─┴┐     ┌┴─┐                  
                  │10│     │11│                  
                  └──┘     └──┘                  
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class JointData(Dataset):
    def __init__(
        self,
        permute_markers=False,
        sequence_interval=128,
        is_trainings_dataset=True,
        base_directory="./",
        apply_mirroring=True,
        cross_validation_index=2,
        k_folds=5,
        supr_path="./data/SUPR/supr_neutral.npy",
        constrained=False,
    ):
        """
        Initializes the JointData dataset.

        Parameters:
        - permute_markers (bool): Whether to apply marker permutation to check transformer behaves as expected.
        - sequence_interval (int): Length of interval for each sequence.
        - is_trainings_dataset (bool): Flag to indicate if the dataset is for training.
        - base_directory (str): Base directory for the dataset.
        - apply_mirroring (bool): Whether to include mirrored sequences.
        - cross_validation_index (int): Index for cross-validation.
        - k_folds (int): Number of folds for cross-validation.
        - supr_path (str): Path to the SUPR model data.
        - constrained (bool): Whether the model should be constrained.
        """

        base_directory_path = base_directory.rstrip("/") + "/"

        # Initialize the SUPR model for processing joint data
        model = SUPR(supr_path, 50)
        model.constrained = constrained
        model.eval()

        # Determine the number of subjects per fold for cross-validation
        sick_per_fold = math.ceil(
            len(os.listdir(f"{base_directory_path}sick")) / k_folds
        )
        healthy_per_fold = math.ceil(
            len(os.listdir(f"{base_directory_path}healthy")) / k_folds
        )

        # Define joint orders for data and their mirrored counterparts
        joint_order = list(range(22))
        mirrored_joint_order = [
            0,
            2,
            1,
            3,
            5,
            4,
            6,
            8,
            7,
            9,
            11,
            10,
            12,
            13,
            14,
            15,
            17,
            16,
            19,
            18,
            21,
            20,
        ]

        # Initialize lists to hold training and evaluation data
        self.num_markers = len(joint_order)
        training_labels, evaluation_labels = [], []
        training_data, evaluation_data = [], []

        # Load joint data from files, organizing into training and evaluation data
        for health_folder in os.listdir(f"{base_directory_path}"):
            health_folder_path = f"{base_directory_path}{health_folder}"
            if not os.path.isdir(health_folder_path):
                continue
            health_status: bool = "healthy" in health_folder
            for index, subject in enumerate(os.listdir(health_folder_path)):
                subject_folder = f"{base_directory_path}{health_folder}/{subject}"
                for gait in os.listdir(subject_folder):
                    gait_folder = f"{subject_folder}/{gait}"
                    poses, betas, translations = [], [], []
                    load_data(
                        data_path=gait_folder,
                        poses=poses,
                        betas=betas,
                        translations=translations,
                    )
                    batch_poses = torch.cat(poses)
                    batch_betas = torch.cat(betas)
                    batch_translations = torch.cat(translations)

                    # Process the data through the SUPR model
                    batched_joints = model.forward(
                        pose=batch_poses, betas=batch_betas, trans=batch_translations
                    ).J_transformed.cpu()

                    # Divide data into training and eval based on cross-validation index
                    if cross_validation_index * (
                        healthy_per_fold if health_status else sick_per_fold
                    ) <= index and index < (cross_validation_index + 1) * (
                        healthy_per_fold if health_status else sick_per_fold
                    ):
                        evaluation_data.append(batched_joints[:, joint_order])
                        evaluation_labels.append(0 if health_status else 1)
                        if apply_mirroring:
                            evaluation_data.append(
                                batched_joints[:, mirrored_joint_order]
                            )
                            evaluation_labels.append(0 if health_status else 1)
                    else:
                        training_data.append(batched_joints[:, joint_order])
                        training_labels.append(0 if health_status else 1)
                        if apply_mirroring:
                            training_data.append(
                                batched_joints[:, mirrored_joint_order]
                            )
                            training_labels.append(0 if health_status else 1)

        # Prepare sequences and labels for training or evaluation
        self.sequences = []
        self.sequence_labels = []
        num_sequences = (
            len(training_data) if is_trainings_dataset else len(evaluation_data)
        )

        for index in range(num_sequences):
            marker_mask = torch.zeros(
                len(training_data[index][0])
                if is_trainings_dataset
                else len(evaluation_data[index][0])
            ).bool()
            marker_id = torch.arange(
                len(training_data[index][0])
                if is_trainings_dataset
                else len(evaluation_data[index][0])
            ).long()
            self.sequences.append(
                (
                    (
                        training_data[index].clone().detach()
                        if is_trainings_dataset
                        else evaluation_data[index].clone().detach()
                    ),
                    marker_id,
                    marker_mask,
                )
            )
        self.sequence_labels = torch.tensor(
            training_labels if is_trainings_dataset else evaluation_labels
        )

        # Apply interval windowing
        new_sequences, new_labels, new_ids, new_masks = (
            [],
            [],
            [],
            [],
        )
        for index in range(len(self.sequences)):
            unfolded_sequence = self.sequences[index][0].unfold(0, sequence_interval, 1)
            interval_list = list(unfolded_sequence.moveaxis(-1, 1))
            new_sequences += interval_list
            new_labels += [self.sequence_labels[index]] * len(interval_list)
            new_ids += [self.sequences[index][1]] * len(interval_list)
            new_masks += [self.sequences[index][2]] * len(interval_list)

        self.sequences = list(zip(new_sequences, new_ids, new_masks))
        self.sequence_labels = new_labels

        # Align, normalize, and possibly permute sequences
        self.align_sequences()
        self.sequences = self.normalize_marker_positions(self.sequences)

        if permute_markers:
            self.apply_permutation()
        else:
            self.augmented_sequences = self.sequences
            self.augmented_labels = self.sequence_labels

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.augmented_sequences)

    def __getitem__(self, index):
        """
        Retrieves the sequence and its label by index.

        Parameters:
        - index (int): Index of the item.

        Returns:
        - tuple: (sequence, label)
        """
        sequence = self.augmented_sequences[index]
        label = self.augmented_labels[index]
        return sequence, torch.squeeze(label)

    def apply_permutation(self, num_permutations=2):
        """
        Augments the dataset by random marker permutations.

        Parameters:
        - num_permutations (int): Number of permutations to apply.
        """
        self.augmented_labels = []
        self.augmented_sequences = []
        for index in range(len(self.sequences)):
            original_sequence = self.sequences[index]
            self.augmented_sequences.append(original_sequence)
            self.augmented_labels.append(self.sequence_labels[index])

            for _ in range(num_permutations):
                random_permutation = torch.randperm(self.num_markers)
                self.augmented_labels.append(self.sequence_labels[index])
                sequence, ids, mask = original_sequence
                self.augmented_sequences.append(
                    (
                        sequence[:, random_permutation, :],
                        ids[random_permutation],
                        mask[random_permutation],
                    )
                )

    def get_healthy_sick_ratio(self):
        """
        Calculates the ratio of healthy to sick labels in the dataset.

        Returns:
        - float: The ratio of healthy to sick participants.
        """
        label_tensor = torch.stack(self.augmented_labels)
        return (label_tensor.shape[0] - label_tensor.sum()) / label_tensor.sum()

    def sort_markers(self):
        """
        Ensures consistent marker sorting across sequences.
        """
        sorted_sequences, sorted_ids, sorted_masks = [], [], []
        for sequence, ids, mask in self.sequences:
            id_map = [-1] * len(ids)  # Initialize ID map
            for idx, id_value in enumerate(ids):
                if id_value != -1:
                    id_map[id_value] = idx
            sorted_sequences.append(sequence[:, id_map, :])
            sorted_ids.append(ids[id_map])
            sorted_masks.append(mask[id_map])
        self.sequences = list(zip(sorted_sequences, sorted_ids, sorted_masks))
        self.augmented_sequences = self.sequences
        self.augmented_labels = self.sequence_labels

    def normalize_marker_positions(self, sequences):
        """
        Normalizes the marker positions in the dataset.

        Parameters:
        - sequences (list): List of sequences to normalize.

        Returns:
        - list: Normalized sequences.
        """
        LANK_ID = 7
        RANK_ID = 8
        HEAD_ID = 15

        normalized_sequences = []
        for sequence, ids, mask in sequences:
            min_values = sequence.view(-1, 3).min(dim=0, keepdim=True)[0]
            max_values = sequence.view(-1, 3).max(dim=0, keepdim=True)[0]

            average_ankle_position = torch.mean(sequence[:, [LANK_ID, RANK_ID], :], 1)
            approximate_height = sequence[:, HEAD_ID, :] - average_ankle_position

            approximate_max_relative_height = torch.max(
                torch.linalg.norm(approximate_height, dim=1)
            )

            # Scale dimensions based on height
            sequence[:, :, 0] /= approximate_max_relative_height

            # Normalize other dimensions
            sequence[:, :, 1:] = (sequence[:, :, 1:] - min_values[:, 1:]) / (
                max_values[:, 1:] - min_values[:, 1:]
            ) * 2 - 1.0

            normalized_sequences.append((sequence, ids, mask))

        return normalized_sequences

    def align_sequences(self):
        """
        Aligns movements in the sequences to a reference orientation.
        """
        LANK_ID = 7
        RANK_ID = 8
        HEAD_ID = 15
        BASE_ID = 0

        for i in range(len(self.sequences)):
            sequence, ids, mask = self.sequences[i]

            # Remove initial offset
            sequence_aligned = sequence - sequence[0][BASE_ID]

            # remove rotation see https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

            # Compute average direction
            average_direction_vector = (
                sequence_aligned[-1][BASE_ID] - sequence_aligned[0][BASE_ID]
            )
            average_direction_vector = torch.nn.functional.normalize(
                average_direction_vector, dim=0
            )
            target_direction_vector = torch.tensor([1, 0, 0], dtype=torch.float)
            cross_product = torch.linalg.cross(
                average_direction_vector, target_direction_vector
            )  # v from stackexchange
            s = torch.linalg.norm(cross_product)
            c = torch.dot(average_direction_vector, target_direction_vector)
            vx_matrix = torch.tensor(
                [
                    [0, -cross_product[2], cross_product[1]],
                    [cross_product[2], 0, -cross_product[0]],
                    [-cross_product[1], cross_product[0], 0],
                ]
            )
            rotation_matrix = (
                torch.eye(3) + vx_matrix + vx_matrix @ vx_matrix * ((1 - c) / (s**2))
            )
            sequence_aligned = torch.matmul(sequence_aligned, rotation_matrix.T)

            # Align upward direction
            current_up_vector = torch.round(
                torch.nn.functional.normalize(
                    sequence_aligned[0][HEAD_ID]
                    - (sequence_aligned[0][RANK_ID] + sequence_aligned[0][LANK_ID]) / 2,
                    dim=0,
                )
            )
            angle = 0
            if abs(current_up_vector[1]) < 1e-4:
                angle = torch.pi / 2 * (-1 if current_up_vector[2] < 0 else 1)
            else:
                if current_up_vector[1] < 1e-4:
                    angle = torch.pi

            angle_tensor = torch.tensor([angle])
            # Round to avoid floating point imprecisions
            rounded_rotation_x = torch.round(
                torch.tensor(
                    [
                        [1, 0, 0],
                        [0, torch.cos(angle_tensor), -torch.sin(angle_tensor)],
                        [0, torch.sin(angle_tensor), torch.cos(angle_tensor)],
                    ]
                )
            )
            sequence_aligned = torch.matmul(sequence_aligned, rounded_rotation_x)
            self.sequences[i] = (sequence_aligned, ids, mask)

        # Ensure left and right consistency across sequences
        for i in range(len(self.sequences)):
            sequence, ids, mask = self.sequences[i]
            if torch.mean(sequence[:, LANK_ID, 2]) > torch.mean(
                sequence[:, RANK_ID, 2]
            ):
                sequence_flipped = sequence * torch.tensor([1, 1, -1])
                self.sequences[i] = (sequence_flipped, ids, mask)
