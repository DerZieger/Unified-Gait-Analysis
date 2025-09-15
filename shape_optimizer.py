from tqdm.autonotebook import tqdm
import torch
import argparse
import os
import json
from util import (
    load_constraints,
    load_markers,
    load_subjects,
    calc_loss,
    save_pose,
)

from supr.pytorch.supr import SUPR
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from adopt import ADOPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    """
    Main function for processing subject data using SUPR and VPoser.
    Loads input data, runs optimization for each frame to align pose data,
    and saves output poses.
    """

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process some parameters")
    parser.add_argument("--cohort", type=str, default="./example_cohort.json")
    parser.add_argument("--save_folder", type=str, default="./out/")
    parser.add_argument(
        "--model_folder",
        type=str,
        default="./data/",
        help="Folder which contains the SUPR and VPoser folders",
    )
    args = parser.parse_args()

    # Extract and prepare directories
    cohort: str = args.cohort
    save_folder: str = args.save_folder.rstrip("/") + "/"
    model_folder: str = args.model_folder.rstrip("/") + "/"
    os.makedirs(save_folder, exist_ok=True)

    # Load subject data from JSON file
    with open(cohort, "r") as json_file:
        json_data = json.load(json_file)
    subjects = load_subjects(json_data)

    # Process each subject
    for (
        subject,
        healthy,
        constraint_file,
        model_file,
        constrained_model,
        gait_file,
    ) in tqdm(subjects, desc="Processing Subjects"):
        healthy_str: str = "healthy" if healthy else "sick"
        constraints = load_constraints(constraint_file)

        # Determine the specific marker constraint
        marker_constraint_name = "Markers"
        for key, value in constraints.items():
            if "c3d" in value["type"]:
                marker_constraint_name = key

        # Load marker positions
        marker_positions = load_markers(
            gait_file, constraints[marker_constraint_name]["data"][3]
        )

        # Initialize SUPR model
        model = SUPR(f"{model_folder}SUPR/{model_file}", 50)
        model.constrained = constrained_model and model.constrained
        model.to(DEVICE)
        model.eval()

        # Initialize VPoser model
        poser = load_model(
            f"{model_folder}/VPoser/",
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )[0]
        poser.to(DEVICE)
        poser.eval()

        batch_size = 1

        # Prepare pose, betas, and translation tensors
        if model.constrained:
            pose = torch.ones(
                batch_size, model.num_pose, device=DEVICE, requires_grad=True
            )
        else:
            poser_in: torch.Tensor = torch.zeros(
                batch_size, 32, device=DEVICE, requires_grad=True
            )
            poser_out: torch.Tensor = poser.decode(poser_in)["pose_body"].reshape(1, -1)
            root = torch.zeros(batch_size, 3, device=DEVICE, requires_grad=True)
            remaining = torch.zeros(
                batch_size,
                model.num_pose - 3 - poser_out.shape[1],
                device=DEVICE,
                requires_grad=True,
            )

        betas: torch.Tensor = torch.zeros(
            batch_size, 50, device=DEVICE, requires_grad=True
        )
        trans: torch.Tensor = torch.zeros(
            batch_size, 3, device=DEVICE, requires_grad=True
        )

        # Process each frame
        for frame in (pbar := tqdm(range(marker_positions.shape[0]), leave=False)):
            # Initialize optimizer and scheduler
            if model.constrained:
                optimizer = torch.optim.Adam(
                    [pose, betas, trans], 0.001, weight_decay=0.01
                )
            else:
                optimizer = torch.optim.Adam(
                    [poser_in, betas, trans, root, remaining], 0.001, weight_decay=0.01
                )

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, threshold=1e-3
            )

            # Optimization loop
            while scheduler.get_last_lr()[0] > 1e-6:
                optimizer.zero_grad()

                if model.constrained:
                    tmp_pose = pose
                else:
                    poser.zero_grad()
                    poser_out = poser.decode(poser_in)["pose_body"].reshape(1, -1)
                    tmp_pose = torch.cat([root, poser_out, remaining], 1)

                model_return: torch.Tensor = model.forward(tmp_pose, betas, trans)
                loss = calc_loss(
                    model.faces,
                    model_return,
                    constraints,
                    tmp_pose,
                    marker_positions[frame],
                )
                loss.backward()
                optimizer.step()
                scheduler.step(loss.item())

            # Save the optimized pose
            gait_name = os.path.basename(gait_file).split(".")[0]
            os.makedirs(
                f"{save_folder}{healthy_str}/{subject}/{gait_name}/", exist_ok=True
            )
            save_pose(
                tmp_pose,
                betas,
                trans,
                f"{save_folder}{healthy_str}/{subject}/{gait_name}/pose_{frame:05d}",
                True,
            )


if __name__ == "__main__":
    main()
