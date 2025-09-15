import torch
import argparse
import imgui
from viewer.mesh import TriMesh, PointCloud, calcNormals
from viewer.shader import Shader
from viewer.window import Window
import glfw
import numpy as np
import time
import os
from util import load_pose, load_data

from supr.pytorch.supr import SUPR
from lrp.core import LRP
from lrp.filter import LayerFilter
from lrp.rules import LrpEpsilonRule
from lrp.zennit.types import AvgPool, Linear, Activation

from typing import Tuple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def align_movement_normalized(joints: torch.Tensor) -> torch.Tensor:
    """
    Aligns movements in the sequences to a reference orientation and normalizes them.

    Parameters:
    - joints (torch.Tensor): A tensor containing joint positions.

    Returns:
    - torch.Tensor: Aligned and normalized joint positions.
    """
    joints = joints[:, list(range(22))]

    LANK_ID = 7
    RANK_ID = 8
    HEAD_ID = 15
    BASE_ID = 0

    # Remove initial offset
    sequence_aligned = joints - joints[0][BASE_ID]

    # remove rotation see https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    # Compute average direction
    average_direction_vector = (
        sequence_aligned[-1][BASE_ID] - sequence_aligned[0][BASE_ID]
    )
    average_direction_vector = torch.nn.functional.normalize(
        average_direction_vector, dim=0
    )
    target_direction_vector = torch.tensor([1, 0, 0], dtype=torch.float).to(
        joints.device
    )
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
    ).to(joints.device)
    rotation_matrix = (
        torch.eye(3).to(joints.device)
        + vx_matrix
        + vx_matrix @ vx_matrix * ((1 - c) / (s**2))
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
    elif current_up_vector[1] < 1e-4:
        angle = torch.pi

    angle_tensor = torch.tensor([angle]).to(joints.device)
    # rounding to ignore small deviation or floating point problems
    rounded_rotation_x = torch.round(
        torch.tensor(
            [
                [1, 0, 0],
                [0, torch.cos(angle_tensor), -torch.sin(angle_tensor)],
                [0, torch.sin(angle_tensor), torch.cos(angle_tensor)],
            ],
        ).to(joints.device)
    )
    sequence_aligned = torch.matmul(sequence_aligned, rounded_rotation_x)

    # Ensure left and right consistency
    if torch.mean(sequence_aligned[:, LANK_ID, 2]) > torch.mean(
        sequence_aligned[:, RANK_ID, 2]
    ):
        sequence_aligned *= torch.tensor([1, 1, -1]).to(joints.device)

    # Normalize sequence
    min_values = sequence_aligned.view(-1, 3).min(dim=0, keepdim=True)[0]
    max_values = sequence_aligned.view(-1, 3).max(dim=0, keepdim=True)[0]

    average_ankle_position = torch.mean(sequence_aligned[:, [LANK_ID, RANK_ID], :], 1)
    approximate_height = sequence_aligned[:, HEAD_ID, :] - average_ankle_position

    approximate_max_relative_height = torch.max(
        torch.linalg.norm(approximate_height, dim=1)
    )

    sequence_aligned[:, :, 0] /= approximate_max_relative_height

    # Normalize other dimensions
    sequence_aligned[:, :, 1:] = (sequence_aligned[:, :, 1:] - min_values[:, 1:]) / (
        max_values[:, 1:] - min_values[:, 1:]
    ) * 2 - 1.0

    return sequence_aligned


def explain_gait_sequence(
    body_model: SUPR,
    poses: list[torch.Tensor],
    betas: list[torch.Tensor],
    translations: list[torch.Tensor],
    network: torch.nn.Module,
) -> Tuple[bool | torch.Tensor]:
    """
    Explain the importance of parts of a gait sequence using a neural network.

    Parameters:
    - body_model (SUPR): The SUPR model used for calculations.
    - poses, betas, translations (list[torch.Tensor]): Pose, beta, and translation tensors.
    - network (torch.nn.Module): The neural network model.

    Returns:
    - Tuple: Classification result and importance values.
    """
    batch_poses: torch.Tensor = torch.cat(poses)
    batch_betas: torch.Tensor = torch.cat(betas)
    batch_translations: torch.Tensor = torch.cat(translations)

    batch_return: torch.Tensor = body_model.forward(
        pose=batch_poses, betas=batch_betas, trans=batch_translations
    )

    batch_joints: torch.Tensor = batch_return.J_transformed
    aligned_joints: torch.Tensor = align_movement_normalized(batch_joints).unsqueeze(0)
    # Transformer or MLP based explanation
    if network.network_type == "transformer":
        pfe_attention_maps, fc_attention_maps = network.get_attnmaps(aligned_joints)
        attn_maps1: torch.Tensor = pfe_attention_maps[0, :, 0, 1:]
        attn_maps2: torch.Tensor = fc_attention_maps[0, 0, 0, 1:]
        importance = torch.einsum("ab,a->ab", attn_maps1, attn_maps2)
        importance /= torch.max(torch.abs(importance))
    else:
        mlp_target_types = (Linear, Activation)
        filter_by_layer_index_type = LayerFilter(
            model=network, target_types=mlp_target_types
        )
        name_map = [
            (
                filter_by_layer_index_type(lambda n: 0 <= n),
                LrpEpsilonRule,
                {"epsilon": 0.25},
            )
        ]

        lrpnet = LRP(network)
        lrpnet.convert_layers(name_map)

        importance = lrpnet.relevance(aligned_joints)[0]

        importance = torch.sum(importance, dim=2)  # sum over xyz for each joint
        # normalize to [-1,1]
        importance /= torch.max(torch.abs(importance))
        importance = (importance + 1) / 2

    return network.forward(aligned_joints) < 0.5, importance


def get_vertex_weights(base_folder: str, model_return: torch.Tensor) -> torch.Tensor:
    """
    Calculate vertex weights for visualization.

    Parameters:
    - base_folder (str): Base folder path to save/load weights.
    - model_return (torch.Tensor): Model tensor to calculate weights from.

    Returns:
    - torch.Tensor: Vertex weights.
    """
    path = f"{base_folder}/vertex_weights.pt"
    if not os.path.isfile(path):
        joints = model_return.J_transformed[0, range(22)]
        distances = torch.cdist(model_return[0], joints)
        closest_joint = torch.min(distances, dim=1)[1]
        closest_distances = torch.zeros_like(distances)
        closest_distances[torch.arange(distances.size(0)), closest_joint] = distances[
            torch.arange(distances.size(0)), closest_joint
        ]  # Only use the distance of the closest joints for each vertex

        non_zero_mask = closest_distances > 0
        min_vals = torch.full(
            (closest_distances.size(1),), float("inf"), device=closest_distances.device
        )
        max_vals = torch.full(
            (closest_distances.size(1),), float("-inf"), device=closest_distances.device
        )

        for joint in range(closest_distances.size(1)):
            non_zero_values = closest_distances[:, joint][non_zero_mask[:, joint]]
            if non_zero_values.numel() > 0:
                min_vals[joint] = torch.min(non_zero_values)
                max_vals[joint] = torch.max(non_zero_values)

        # Normalize each column individually
        range_vals = max_vals - min_vals + 1e-7  # Avoid division by zero
        vertex_weights = torch.where(
            non_zero_mask,
            1 - (closest_distances - min_vals) / range_vals,
            closest_distances,
        )  # Normalize the distance over greater than zero and invert to get interpolation weight
        torch.save(vertex_weights, path)
    else:
        vertex_weights = torch.load(path, weights_only=True)
    return vertex_weights


def main():
    """
    Main function for the script execution.
    Sets up the OpenGL window, loads shaders, models, and handles the UI for interaction.
    """
    parser = argparse.ArgumentParser(description="Process some parameters")
    parser.add_argument(
        "--base_folder",
        type=str,
        default="./data/SUPR",
        help="Folder which contains the SUPR files",
    )
    parser.add_argument("--model", type=str, default="supr_neutral.npy")
    parser.add_argument("--unconstrained", action="store_true")
    args = parser.parse_args()
    base_folder = args.base_folder.rstrip("/") + "/"
    model_path = args.model
    constrained = not args.unconstrained

    win = Window(1920, 1080, "Test")

    # Load shaders and create shader program
    bodyshader = Shader("viewer/simple.vert", "viewer/body.frag")
    bodyshader.uniformInt("colormap", int(0))

    # Start imgui frame
    imgui.new_frame()
    start = time.time()

    model = SUPR(f"{base_folder}{model_path}", 50)
    model.constrained = constrained
    model.to(DEVICE)
    model.eval()

    batch_size = 1
    poses: list[torch.Tensor] = [torch.zeros(batch_size, model.num_pose, device=DEVICE)]
    betas: list[torch.Tensor] = [torch.zeros(batch_size, 50, device=DEVICE)]
    translations: list[torch.Tensor] = [torch.zeros(batch_size, 3, device=DEVICE)]
    model_return: torch.Tensor = model.forward(poses[0], betas[0], translations[0])

    model_mesh = TriMesh(
        model_return.detach().cpu().numpy(),
        model.f,
        calcNormals(model_return.detach().cpu().numpy()[0], model.f),
        bodyshader,
        True,
    )
    model_mesh.updateVertCol(np.ones((model_mesh.numVerts, 4), dtype=np.float32))
    bodyshader.uniformInt("colormap", -1)

    vertex_weights = get_vertex_weights(
        base_folder=base_folder, model_return=model_return
    )

    # Imgui variables
    network_path = "NN PATH HERE"
    data_path = "PATH TO POSE FOLDER"
    display_model_query = True
    display_data_query = True
    display_xai_query = 0
    show_xai = False
    importances = torch.ones([1])
    healty = False
    explain_start_frame = 0
    failed_load = False
    current_frame = 0
    classified_from = 0
    classified_to = 0

    # Main application loop
    while not glfw.window_should_close(win.window):
        win.newFrame()
        model_mesh.render(win.camera, False)

        if imgui.begin("Controlpanel"):
            imgui.text(f"Frame time: {(time.time() - start):.04f}")
            start = time.time()
            changed = False

            imgui.push_id("model")
            if display_model_query:
                imgui.text("Model path: ")
                imgui.same_line()
                _, network_path = imgui.input_text("", network_path)
                if imgui.button("Load model"):
                    if os.path.isfile(network_path) and ".ckpt" in network_path:
                        display_model_query = False
                        failed_load = False
                        display_xai_query += 1
                        network = torch.load(network_path, weights_only=False)
                    else:
                        failed_load = True
                if failed_load:
                    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.0, 0.0)
                    imgui.text("No valid model")
                    imgui.pop_style_color(1)
            else:
                imgui.text(f"Model already loaded, type: {network.network_type}")
                imgui.same_line()
                if imgui.button("Remove model"):
                    display_model_query = True
                    display_xai_query -= 1
            imgui.pop_id()

            imgui.push_id("data")
            if display_data_query:
                imgui.text("Data path: ")
                imgui.same_line()
                _, data_path = imgui.input_text("", data_path)
                if imgui.button("Load data"):
                    if os.path.isdir(data_path):
                        display_data_query = False
                        failed_load = False
                        display_xai_query += 1
                        load_data(
                            data_path=data_path,
                            poses=poses,
                            betas=betas,
                            translations=translations,
                        )
                    else:
                        failed_load = True
                if failed_load:
                    imgui.push_style_color(imgui.COLOR_TEXT, 1.0, 0.0, 0.0)
                    imgui.text("No valid data")
                    imgui.pop_style_color(1)
            else:
                imgui.text("Data already loaded")
                imgui.same_line()
                if imgui.button("Remove data"):
                    display_data_query = True
                    display_xai_query -= 1
                    poses: list[torch.Tensor] = [
                        torch.zeros(batch_size, model.num_pose, device=DEVICE)
                    ]
                    betas: list[torch.Tensor] = [
                        torch.zeros(batch_size, 50, device=DEVICE)
                    ]
                    translations: list[torch.Tensor] = [
                        torch.zeros(batch_size, 3, device=DEVICE)
                    ]
                    current_frame = 0
            imgui.pop_id()

            changed, current_frame = imgui.slider_int(
                "Current frame",
                current_frame,
                min_value=0,
                max_value=len(poses) - 1,
                format="%d",
            )
            if changed:
                model_return: torch.Tensor = model.forward(
                    poses[current_frame],
                    betas[current_frame],
                    translations[current_frame],
                )
                model_mesh.updateVerts(model_return.detach().cpu().numpy())
                model_mesh.updateNorms(
                    calcNormals(model_return.detach().cpu().numpy()[0], model.f)
                )
                if show_xai:
                    current_importance = importances[
                        max(
                            min(
                                current_frame - explain_start_frame,
                                network.interval_length - 1,
                            ),
                            0,
                        )
                    ]
                    if network.network_type == "transformer":
                        current_importance = torch.einsum(
                            "ab,b->a", vertex_weights, current_importance
                        )
                    elif network.network_type == "lrpmlp":
                        interpolation_factor = torch.einsum(
                            "ab,b->a", vertex_weights, torch.ones([22]).to(DEVICE)
                        )
                        current_importance = (
                            1 - interpolation_factor
                        ) * 0.5 * torch.ones([10475]).to(DEVICE) + torch.einsum(
                            "ab,b->a", vertex_weights, current_importance
                        )
                    model_mesh.updateVertCol(
                        torch.stack(
                            [
                                current_importance,
                                current_importance,
                                current_importance,
                                current_importance,
                            ],
                            1,
                        )
                        .contiguous()
                        .detach()
                        .cpu()
                        .numpy()
                    )

            imgui.push_id("xai")

            expanded, _ = imgui.collapsing_header("Explainable Gait", None)
            if expanded:
                if (
                    display_xai_query > 1
                    and network.network_type
                    in [
                        "transformer",
                        "lrpmlp",
                    ]
                    and network.interval_length < len(poses)
                ):
                    imgui.text(
                        f"Explainable sequence length: {network.interval_length}, gait sequence length: {len(poses)}"
                    )

                    _, explain_start_frame = imgui.slider_int(
                        "Explain sequence start frame",
                        explain_start_frame,
                        min_value=0,
                        max_value=len(poses) - network.interval_length,
                        format="%d",
                    )

                    if imgui.button("Explain gait sequence"):
                        healty, importances = explain_gait_sequence(
                            model,
                            poses=poses[
                                explain_start_frame : explain_start_frame
                                + network.interval_length
                            ],
                            betas=betas[
                                explain_start_frame : explain_start_frame
                                + network.interval_length
                            ],
                            translations=translations[
                                explain_start_frame : explain_start_frame
                                + network.interval_length
                            ],
                            network=network,
                        )
                        show_xai = True
                        bodyshader.uniformInt(
                            "colormap",
                            0 if network.network_type == "transformer" else 1,
                        )
                        classified_from = explain_start_frame
                        classified_to = explain_start_frame + network.interval_length

                else:
                    imgui.text(
                        "First load a long enough gait sequence and compatible network"
                    )
                if show_xai:
                    healty_text = "healthy" if healty else "sick"
                    imgui.text(
                        f"The sequence between timestep {classified_from} and {classified_to} was classified as {healty_text}"
                    )
                    imgui.text(
                        f"The visualize importances are only valid between timestep {classified_from} and {classified_to}"
                    )

            imgui.pop_id()

            imgui.end()

    win.imgui_impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
