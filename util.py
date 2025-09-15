import torch
import os
import json
from ezc3d import c3d
import random
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_pose(filename: str):
    """
    Load pose, beta, and translation data from a file.

    Parameters:
    - filename (str): The path to the file to load from.

    Returns:
    - Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing betas, translation, and pose tensors.
    """
    with open(filename) as pose_file:
        beta, translation, pose = [], [], []
        if ".txt" in filename:
            content = pose_file.read()
            content_float = [float(i) for i in content.split(" ") if i.strip()]
            beta = [content_float[-50:]]
            translation = [content_float[3:6]]
            pose = [content_float[:3] + content_float[6:-50]]
        elif ".json" in filename:
            content = json.load(pose_file)
            beta = content["betas"]
            pose = content["pose"]
            translation = content["translation"]
        beta = torch.tensor(beta, device=DEVICE)
        translation = torch.tensor(translation, device=DEVICE)
        pose = torch.tensor(pose, device=DEVICE)
        return beta, translation, pose


def save_pose(
    pose: torch.Tensor,
    betas: torch.Tensor,
    translation: torch.Tensor,
    filename: str = "./placeholder.json",
    json_only: bool = True,
):
    """
    Save pose, betas, and translation to JSON and optionally text format.

    Parameters:
    - pose (torch.Tensor): The pose tensor to save.
    - betas (torch.Tensor): The beta tensor to save.
    - translation (torch.Tensor): The translation tensor to save.
    - filename (str): The base filename to save to.
    - json_only (bool): If True, only save in JSON format.
    """
    directory = os.path.dirname(filename)
    os.makedirs(directory, exist_ok=True)
    pose_out = {
        "translation": translation.contiguous().cpu().tolist(),
        "betas": betas.contiguous().cpu().tolist(),
        "pose": pose.contiguous().cpu().tolist(),
    }
    with open(filename + ".json", "w") as outfile:
        json.dump(pose_out, outfile, indent=4, sort_keys=True)
    if not json_only:
        with open(filename + ".txt", "w") as f:
            for v in pose_out["pose"][0][:3]:
                f.write(f"{v} ")
            for v in pose_out["translation"][0]:
                f.write(f"{v} ")
            for v in pose_out["pose"][0][3:]:
                f.write(f"{v} ")
            for v in pose_out["betas"][0]:
                f.write(f"{v} ")


def load_constraints(path: str):
    """
    Load constraints from a JSON file into a structured dictionary.

    Parameters:
    - path (str): The path to the constraints file.

    Returns:
    - Dict: A dictionary containing constraint data.
    """
    constraints_opt = {}
    with open(path, "r") as f:
        constraints = json.load(f)
        for k, v in constraints.items():
            constraints_opt[k] = {}
            if "C3DMarker" in v["type"]:
                distances, faceIds, barycenterCoords, marker_names = [], [], [], []
                for marker_name, marker_meta in v["Markers"].items():
                    distances.append(marker_meta["distanceToSkin"])
                    faceIds.append(marker_meta["faceId"])
                    barycenterCoords.append(marker_meta["baryCoords"])
                    marker_names.append(marker_name)
                constraints_opt[k]["type"] = "c3d"
                constraints_opt[k]["data"] = (
                    torch.tensor(distances, device=DEVICE),
                    torch.tensor(faceIds, dtype=torch.int, device=DEVICE),
                    torch.tensor(barycenterCoords, device=DEVICE),
                    marker_names,
                )
            if "Angle" in v["type"]:
                joint_ids = v["joints"]
                lower_bounds = v["lbounds"]
                upper_bounds = v["upbounds"]
                constraints_opt[k]["type"] = "angle"
                constraints_opt[k]["data"] = (
                    torch.tensor(joint_ids, dtype=torch.int, device=DEVICE),
                    torch.tensor(lower_bounds, device=DEVICE),
                    torch.tensor(upper_bounds, device=DEVICE),
                )
    return constraints_opt


def load_markers(path: str, marker_names: list):
    """
    Load marker positions from a C3D file and convert to metric units.

    Parameters:
    - path (str): The path to the C3D file.
    - marker_names (list): List of marker names to load.

    Returns:
    - torch.Tensor: A tensor of the marker positions.
    """
    reader = c3d(path)
    point_labels = reader["parameters"]["POINT"]["LABELS"]["value"]

    marker_position = []
    for name in marker_names:
        if name in point_labels:
            marker_position.append(point_labels.index(name))

    tmp_tensor = torch.tensor(
        reader["data"]["points"][:3], device=DEVICE, dtype=torch.float
    ).transpose(2, 0)
    conversion = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1.0}

    tmp_tensor = tmp_tensor[:, marker_position, :]
    return tmp_tensor * conversion[reader["parameters"]["POINT"]["UNITS"]["value"][0]]


def calc_loss(faces, v_posed, constraints, pose, positions):
    """
    Calculate the loss for the pose given constraints and marker positions.

    Parameters:
    - faces: Face indices for the mesh.
    - v_posed: Posed vertices for the mesh.
    - constraints: A dictionary containing constraint data.
    - pose: The current pose tensor.
    - positions: Marker positions to match.

    Returns:
    - torch.Tensor: The calculated loss.
    """
    sum = torch.zeros(1, device=DEVICE)
    for key, value in constraints.items():
        if "c3d" in value["type"]:
            distances, faceIds, barycoords, _ = value["data"]
            vi = faces[faceIds]
            vs = v_posed[:, vi]
            diff0 = vs[:, :, 1] - vs[:, :, 0]
            diff1 = vs[:, :, 2] - vs[:, :, 0]
            normal = torch.linalg.cross(diff0, diff1)
            normaln = torch.nn.functional.normalize(normal)
            point = torch.einsum("bckd,ck->bcd", [vs, barycoords])
            point2 = point + normaln * distances[None, :, None]

            dist = point2 - positions
            res = torch.linalg.norm(dist, dim=2)

            sum += torch.sum(res, 1)

        if "angle" in value["type"]:
            joint_ids, lower_bounds, upper_bounds = value["data"]

            pose_values = pose[0, joint_ids]
            error = torch.where(
                pose_values > upper_bounds,
                torch.pow(pose_values - upper_bounds, 2) * 2,
                0,
            )
            error = torch.where(
                pose_values < lower_bounds,
                torch.pow(pose_values - lower_bounds, 2) * 2,
                error,
            )

            sum += torch.sum(error)
    return sum


def seed_randomness(seed: int = 0):
    """
    Seed all possible sources of randomness to obtain deterministic results.

    Parameters:
    - seed (int): The seed value to use for random number generation.

    Returns:
    - torch.Generator: A new random number generator with the specified seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def load_data(
    data_path: str,
    poses: list[torch.Tensor],
    betas: list[torch.Tensor],
    translations: list[torch.Tensor],
):
    """
    Load pose, beta, and translation data from files in a directory.

    Parameters:
    - data_path (str): Directory path containing pose files.
    - poses (list[torch.Tensor]): List to append loaded pose tensors.
    - betas (list[torch.Tensor]): List to append loaded beta tensors.
    - translations (list[torch.Tensor]): List to append loaded translation tensors.
    """
    paths = os.listdir(data_path)
    paths.sort()
    betas.clear()
    translations.clear()
    poses.clear()
    for pose_path in paths:
        beta, translation, pose = load_pose(os.path.join(data_path, pose_path))
        betas.append(beta)
        translations.append(translation)
        poses.append(pose)


def load_subjects(scene_json):
    """
    Load subject information from scene JSON data.

    Parameters:
    - scene_json: JSON data containing scene and subject information.

    Returns:
    - List[Tuple]: A list of tuples containing subject data.
    """
    subjects = []
    for name, data in scene_json.items():
        for gait in data["gaits"]:
            subjects.append(
                (
                    name,
                    data["healthy"],
                    data["constraint_file"],
                    data["model_file"],
                    data["constrained_model"],
                    gait,
                )
            )
    return subjects
