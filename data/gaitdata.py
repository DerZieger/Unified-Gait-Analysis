from torch.utils.data import Dataset
import json
import torch
from ezc3d import c3d
import numpy as np
import os
import math
from util import load_subjects


class ParkGait(Dataset):
    def __init__(
        self,
        cohort_file,
        permute_markers=False,
        sequence_interval=128,
        is_trainings_dataset=True,
        dictionary=None,
        apply_mirroring=True,
        cross_validation_index=2,
        k_fold=5,
    ):
        """
        Initializes the Parkinson Gait dataset.

        Parameters:
        - cohort_file (str): Path to file listing all subjects.
        - permute_markers (bool): Whether to apply marker permutation to check transformer behaves as expected.
        - sequence_interval (int): Length of interval for sequence windowing.
        - is_trainings_dataset (bool): Indicates if the dataset is for training.
        - dictionary (dict): Mapping of marker names to indices.
        - apply_mirroring (bool): Whether to include mirrored sequences.
        - cross_validation_index (int): Index for cross-validation splitting.
        - k_fold (int): Number of folds for cross-validation.
        """

        # List of marker names used in the dataset
        marker_names = [
            "RFHD",
            "LFHD",
            "LBHD",
            "RBHD",
            "RSHO",
            "LSHO",
            "CLAV",
            "STRN",
            "LUPA",
            "RUPA",
            "LELB",
            "RELB",
            "LFRM",
            "RFRM",
            "LASI",
            "RASI",
            "RWRA",
            "RWRB",
            "LWRA",
            "LWRB",
            "RFIN",
            "LFIN",
            "RTHAP",
            "LTHAP",
            "RTHI",
            "LTHI",
            "RTAD",
            "LTAD",
            "RKNE",
            "RKNM",
            "LKNE",
            "LKNM",
            "RTIAP",
            "LTIAP",
            "RTIB",
            "LTIB",
            "LTIAD",
            "RTIAD",
            "RANK",
            "LANK",
            "RMED",
            "LMED",
            "RTOE",
            "LTOE",
            "C7",
            "RBAK",
            "T10",
            "RPSI",
            "LPSI",
            "RHEE",
            "LHEE",
            "LKAX",
            "RKAX",
            "RKD1",
            "RKD2",
            "LKD1",
            "LKD2",
            "SCAR",
            "SACR",
            "SHO",
            "ULN",
            "RAD",
            "W1",
            "W2",
            "W3",
            "W4",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "M1",
            "M2",
            "M3",
            "M4",
            "M5",
            "I1",
            "I2",
            "I3",
            "I4",
            "I5",
            "T1",
            "T2",
            "T3",
            "T4",
            "T5",
            "T8",
        ]

        with open(cohort_file, "r") as json_file:
            json_data = json.load(json_file)

        # Load all subject data
        subjects = load_subjects(json_data)

        number_healthy = 0  # Counter for healthy subjects
        number_sick = 0  # Counter for sick subjects

        # Count healthy and sick occurrences
        for name, data in json_data.items():
            for gait in data["gaits"]:
                number_healthy += 1 if data["healthy"] else 0
                number_sick += 0 if data["healthy"] else 1

        # Calculate number of subjects per fold for cross-validation
        sick_per_fold = math.ceil(number_sick / k_fold)
        healthy_per_fold = math.ceil(number_healthy / k_fold)

        # Initialize marker dictionary
        self.dictionary = {} if dictionary is None else dictionary

        training_labels, evaluation_labels = [], []
        training_data, evaluation_data = [], []
        self.num_markers = len(self.dictionary)

        # Process each subject and gather training/evaluation data
        health_index = 0
        sick_index = 0
        for subject_index, data in enumerate(json_data.items()):
            subject_name, subject_data = data
            health_status = subject_data["healthy"]
            index = health_index if health_status else sick_index
            for gait in subject_data["gaits"]:
                if not os.path.isfile(gait):
                    continue
                c3d_data = c3d(gait)
                # Obtain point labels from c3d data
                point_labels = [
                    pointlabel.split(":")[-1]
                    for pointlabel in c3d_data["parameters"]["POINT"]["LABELS"]["value"]
                ]

                # Get the indices of marker points
                marker_indices = [
                    label_idx
                    for label_idx, label in enumerate(point_labels)
                    if label.strip() in marker_names
                ]

                # Update the marker dictionary and count number of markers
                for label in (point_labels[idx].strip() for idx in marker_indices):
                    if label not in self.dictionary:
                        self.dictionary[label] = self.num_markers
                        self.num_markers += 1

                local_marker_indices = torch.tensor(marker_indices).to(torch.int)
                global_marker_indices = torch.tensor(
                    [
                        self.dictionary[label.strip()]
                        for label in point_labels
                        if label in marker_names
                    ]
                ).to(torch.int)

                # Gather sequence data for current walk
                current_walk = []
                for frame in range(len(c3d_data["data"]["points"][0][0])):
                    # Use first three columns of 'points' for all frames
                    frame_data = torch.tensor(
                        np.stack(
                            (
                                c3d_data["data"]["points"][
                                    0, local_marker_indices, frame
                                ],
                                c3d_data["data"]["points"][
                                    1, local_marker_indices, frame
                                ],
                                c3d_data["data"]["points"][
                                    2, local_marker_indices, frame
                                ],
                            ),
                            axis=1,
                        )
                    )
                    current_walk.append(frame_data)

                # Skip sequences with NaN values
                if torch.isnan(torch.stack(current_walk)).any():
                    continue

                # Split data into training and evaluation based on cross-validation index
                if cross_validation_index * (
                    healthy_per_fold if health_status else sick_per_fold
                ) <= index and index < (cross_validation_index + 1) * (
                    healthy_per_fold if health_status else sick_per_fold
                ):
                    evaluation_data.append(
                        (torch.stack(current_walk), global_marker_indices)
                    )
                    evaluation_labels.append(0 if health_status else 1)
                else:
                    training_data.append(
                        (torch.stack(current_walk), global_marker_indices)
                    )
                    training_labels.append(0 if health_status else 1)
            health_index += 1 if health_status else 0
            sick_index += 0 if health_status else 1

        if is_trainings_dataset:
            data, labels = training_data, training_labels
        else:
            data, labels = evaluation_data, evaluation_labels

        # Align and format sequences
        self.sequences = []
        self.seq_labels = labels
        for sequence, marker_indices in data:
            marker_indices = np.pad(
                marker_indices,
                (0, self.num_markers - len(marker_indices)),
                mode="constant",
                constant_values=-1,
            )
            mask = marker_indices == -1
            self.sequences.append(
                (
                    torch.tensor(
                        np.pad(
                            sequence,
                            ((0, 0), (0, self.num_markers - sequence.shape[1]), (0, 0)),
                        )
                    ),
                    torch.tensor(marker_indices).long(),
                    torch.tensor(mask).bool(),
                )
            )

        # Prepare sequences for windowing
        windowed_sequences, windowed_labels, windowed_indices, windowed_masks = (
            [],
            [],
            [],
            [],
        )
        for i in range(len(self.sequences)):
            windowed = self.sequences[i][0].unfold(0, sequence_interval, 1)
            list_of_intervals = list(windowed.moveaxis(-1, 1))
            windowed_sequences += list_of_intervals
            windowed_labels += [self.seq_labels[i]] * len(list_of_intervals)
            windowed_indices += [self.sequences[i][1]] * len(list_of_intervals)
            windowed_masks += [self.sequences[i][2]] * len(list_of_intervals)

        self.sequences = list(zip(windowed_sequences, windowed_indices, windowed_masks))
        self.seq_labels = windowed_labels

        # Apply mirroring if required
        if apply_mirroring:
            self.sequences, self.seq_labels = self.mirror_augmentation(
                self.sequences, self.seq_labels
            )

        self.align_movement()  # Align movements for uniformity
        self.sequences = self.normalize_marker_positions(
            self.sequences
        )  # Normalize markers

        # Augment the data if required
        if permute_markers:
            self.apply_permutation()
        else:
            self.augments = self.sequences
            self.aug_labels = self.seq_labels

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        return len(self.augments)

    def __getitem__(self, idx):
        """
        Retrieves a sequence and its associated label by index.

        Parameters:
        - idx (int): Index of the desired sequence.

        Returns:
        - tuple: (sequence, label)
        """
        sequence = self.augments[idx]
        label = self.aug_labels[idx]
        return sequence, torch.squeeze(label)

    def getDictionary(self):
        """
        Returns the marker dictionary used in the dataset.

        Returns:
        - dict: The marker dictionary.
        """
        return self.dictionary

    def apply_permutation(self, permutations=2):
        """
        Augments the dataset by applying random permutations to marker positions.

        Parameters:
        - permutations (int): Number of permutations to apply.
        """
        self.aug_labels = []
        self.augments = []
        for idx in range(len(self.sequences)):
            original_sequence = self.sequences[idx]
            self.augments.append(original_sequence)
            self.aug_labels.append(self.seq_labels[idx])

            for _ in range(permutations):
                # Apply marker permutation to sequence
                permutation = torch.randperm(self.num_markers)
                self.aug_labels.append(self.seq_labels[idx])
                sequence, markers, mask = original_sequence
                self.augments.append(
                    (
                        sequence[:, permutation, :],
                        markers[permutation],
                        mask[permutation],
                    )
                )

    def get_healthy_sick_ratio(self):
        """
        Calculates the ratio of healthy to sick participants in the dataset.

        Returns:
        - float: The ratio of healthy to sick participants.
        """
        labels = torch.stack(self.aug_labels)
        return (labels.shape[0] - labels.sum()) / labels.sum()

    def sortMarker(self):
        """
        Ensures consistent marker sorting across all sequences.
        """
        sorted_sequences, sorted_indices, sorted_masks = [], [], []
        for sequence, indices, mask in self.sequences:
            id_map = [-1] * len(indices)
            for index in range(len(indices)):
                if indices[index] != -1:
                    id_map[indices[index]] = index
            # Sort sequences and update indices and masks
            sorted_sequences.append(sequence[:, id_map, :])
            sorted_indices.append(indices[id_map])
            sorted_masks.append(mask[id_map])
        self.sequences = list(zip(sorted_sequences, sorted_indices, sorted_masks))
        self.augments = self.sequences
        self.aug_labels = self.seq_labels

    def align_movement(self):
        """
        Aligns movements in sequences to a reference orientation.
        """
        for i in range(len(self.sequences)):
            seq, ids, mask = self.sequences[i]
            left_ankle_idx = self.find_index_for_marker_in_ids("LANK", ids)
            right_ankle_idx = self.find_index_for_marker_in_ids("RANK", ids)
            left_front_head_idx = self.find_index_for_marker_in_ids("LFHD", ids)

            # Remove initial offset based on C7 marker
            aligned_seq = seq - seq[0][self.find_index_for_marker_in_ids("C7", ids)]

            # Calculate the rotation matrix to align the average forward movement to the x-axis
            average_direction = (
                torch.mean(aligned_seq[-1], 0, True)[0]
                - torch.mean(aligned_seq[0], 0, True)[0]
            )
            average_direction = torch.nn.functional.normalize(average_direction, dim=0)
            target_direction = torch.tensor([1, 0, 0], dtype=torch.float)
            cross_product = torch.linalg.cross(average_direction, target_direction)
            scalar_product = torch.linalg.norm(cross_product)
            dot_product = torch.dot(average_direction, target_direction)
            skew_symmetric_matrix = torch.tensor(
                [
                    [0, -cross_product[2], cross_product[1]],
                    [cross_product[2], 0, -cross_product[0]],
                    [-cross_product[1], cross_product[0], 0],
                ]
            )
            rotation_matrix = (
                torch.eye(3)
                + skew_symmetric_matrix
                + (skew_symmetric_matrix @ skew_symmetric_matrix)
                * ((1 - dot_product) / (scalar_product**2))
            )
            aligned_seq = torch.matmul(aligned_seq, rotation_matrix.T)

            # Align the upward direction (y-axis alignment)
            upward_vector = torch.round(
                torch.nn.functional.normalize(
                    aligned_seq[0][left_front_head_idx]
                    - (aligned_seq[0][right_ankle_idx] + aligned_seq[0][left_ankle_idx])
                    / 2,
                    dim=0,
                )
            )
            alpha_angle = 0
            if abs(upward_vector[1]) < 1e-4:
                alpha_angle = torch.pi / 2 * (-1 if upward_vector[2] < 0 else 1)
            elif upward_vector[1] < 1e-4:
                alpha_angle = torch.pi

            alpha_tensor = torch.tensor([alpha_angle])
            rotation_x = torch.round(
                torch.tensor(
                    [
                        [1, 0, 0],
                        [0, torch.cos(alpha_tensor), -torch.sin(alpha_tensor)],
                        [0, torch.sin(alpha_tensor), torch.cos(alpha_tensor)],
                    ]
                )
            )
            aligned_seq = torch.matmul(aligned_seq, rotation_x)
            self.sequences[i] = (aligned_seq, ids, mask)

        # Ensure left-right consistency
        for i in range(len(self.sequences)):
            seq, ids, mask = self.sequences[i]
            left_ankle_idx = self.find_index_for_marker_in_ids("LANK", ids)
            right_ankle_idx = self.find_index_for_marker_in_ids("RANK", ids)
            # Mirror sequence if necessary to ensure consistency
            if torch.mean(seq[:, left_ankle_idx, 2]) > torch.mean(
                seq[:, right_ankle_idx, 2]
            ):
                mirrored_seq = seq * torch.tensor([1, 1, -1])
                self.sequences[i] = (mirrored_seq, ids, mask)

    def normalize_marker_positions(self, sequences):
        """
        Normalizes the positions of markers within sequences.

        Parameters:
        - sequences (list): A list of sequences containing marker positions.

        Returns:
        - list: The normalized sequences.
        """
        normalized_sequences = []
        for seq, marker_ids, mask in sequences:

            left_ankle_idx = self.find_index_for_marker_in_ids("LANK", marker_ids)
            right_ankle_idx = self.find_index_for_marker_in_ids("RANK", marker_ids)
            head_idx = self.find_index_for_marker_in_ids("RFHD", marker_ids)
            min_vec = seq.view(-1, 3).min(dim=0, keepdim=True)[0]
            max_vec = seq.view(-1, 3).max(dim=0, keepdim=True)[0]

            mean_ankle_position = torch.mean(
                seq[:, [left_ankle_idx, right_ankle_idx], :], 1
            )
            approx_heights = seq[:, head_idx, :] - mean_ankle_position

            max_relative_height = torch.max(torch.linalg.norm(approx_heights, dim=1))

            # Scale x-dimension by participant height
            seq[:, :, 0] /= max_relative_height

            # Normalize y and z dimensions
            seq[:, :, 1:] = (
                (seq[:, :, 1:] - min_vec[:, 1:]) / (max_vec[:, 1:] - min_vec[:, 1:])
            ) * 2 - 1.0

            normalized_sequences.append((seq, marker_ids, mask))

        return normalized_sequences

    def find_index_for_marker_in_ids(self, marker_name, marker_ids):
        """
        Finds the index of a given marker within the marker IDs.

        Parameters:
        - marker_name (str): The name of the marker.
        - marker_ids (array-like): The array of marker identifiers.

        Returns:
        - int: The index of the marker, or -1 if not found.
        """
        if marker_name in self.dictionary:
            if self.dictionary[marker_name] in marker_ids:
                return (
                    (marker_ids == self.dictionary[marker_name])
                    .nonzero(as_tuple=True)[0][0]
                    .item()
                )
        return -1

    def mirror_augmentation(self, sequences, labels):
        """
        Performs mirror augmentation on sequences.

        Parameters:
        - sequences (list): List of sequences.
        - labels (list): List of corresponding sequence labels.

        Returns:
        - tuple: Augmented sequences and labels.
        """
        augmented_sequences = []
        augmented_labels = []
        marker_swaps = [
            ("RFHD", "LFHD"),
            ("LBHD", "RBHD"),
            ("RSHO", "LSHO"),
            ("LUPA", "RUPA"),
            ("LELB", "RELB"),
            ("LFRM", "RFRM"),
            ("LASI", "RASI"),
            ("RWRA", "LWRA"),
            ("RWRB", "LWRB"),
            ("RFIN", "LFIN"),
            ("RTHAP", "LTHAP"),
            ("RTHI", "LTHI"),
            ("RTAD", "LTAD"),
            ("RKNE", "LKNE"),
            ("RKNM", "LKNM"),
            ("RTIAP", "LTIAP"),
            ("RTIB", "LTIB"),
            ("LTIAD", "RTIAD"),
            ("RANK", "LANK"),
            ("RMED", "LMED"),
            ("RTOE", "LTOE"),
            ("RPSI", "LPSI"),
            ("RHEE", "LHEE"),
            ("LKAX", "RKAX"),
            ("RKD1", "LKD1"),
            ("RKD2", "LKD2"),
        ]

        for seq_idx in range(len(sequences)):
            # Add unchanged sequence
            augmented_labels.append(labels[seq_idx].item())
            augmented_sequences.append(sequences[seq_idx])

            # Add mirrored sequence
            augmented_labels.append(labels[seq_idx].item())
            current_seq, marker_ids, mask = sequences[seq_idx]

            new_marker_ids = marker_ids.detach().clone()
            new_mask = mask.detach().clone()
            for marker1, marker2 in marker_swaps:
                idx_marker1 = self.find_index_for_marker_in_ids(marker1, marker_ids)
                idx_marker2 = self.find_index_for_marker_in_ids(marker2, marker_ids)

                if idx_marker1 == -1 and idx_marker2 == -1:
                    continue
                if idx_marker1 == -1:
                    new_mask[idx_marker2] = True
                    new_marker_ids[idx_marker2] = -1
                    continue
                if idx_marker2 == -1:
                    new_mask[idx_marker1] = True
                    new_marker_ids[idx_marker1] = -1
                    continue

                # Swap the indices and masks
                new_mask[idx_marker1] = mask[idx_marker2]
                new_mask[idx_marker2] = mask[idx_marker1]
                new_marker_ids[idx_marker1] = marker_ids[idx_marker2]
                new_marker_ids[idx_marker2] = marker_ids[idx_marker1]

            augmented_sequences.append((current_seq, new_marker_ids, new_mask))

        # Convert labels to tensors
        augmented_labels = [
            torch.tensor([label]).to(torch.long) for label in augmented_labels
        ]
        return augmented_sequences, augmented_labels
