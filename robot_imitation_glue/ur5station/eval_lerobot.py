import os

import numpy as np
import torch
from torchvision import transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.gello.gello_agent import GelloAgent
from robot_imitation_glue.agents.lerobot_agent import LerobotAgent, make_lerobot_policy
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.ur5station.data_collection import policy_action_to_abs_se3_converter
from robot_imitation_glue.ur5station.ur5_robot_env import (
    UR5eStation,
    convert_abs_gello_actions_to_se3,
    dynamixel_config,
)

if __name__ == "__main__":
    checkpoint_path = "/home/tlips/Code/robot-imitation-glue/outputs/train/2025-04-10/13-15-24_pick-cube_diffusion/checkpoints/035000/pretrained_model"
    dataset_path = "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-remapped"

    def preprocessor(obs_dict):
        scene_img = obs_dict["scene_image"]
        wrist_img = obs_dict["wrist_image"]
        state = obs_dict["state"]

        state = torch.tensor(state).float().unsqueeze(0)
        scene_image = torch.tensor(scene_img).float() / 255.0
        wrist_image = torch.tensor(wrist_img).float() / 255.0
        scene_image = scene_image.permute(2, 0, 1)
        wrist_image = wrist_image.permute(2, 0, 1)

        transform = transforms.Compose([transforms.CenterCrop(196)])
        scene_image = transform(scene_image)
        wrist_image = transform(wrist_image)

        # unsqueeze images
        scene_image = scene_image.unsqueeze(0)
        wrist_image = wrist_image.unsqueeze(0)

        return {
            "observation.images.scene_image": scene_image,
            "observation.images.wrist_image": wrist_image,
            "observation.state": state,
        }

    env = UR5eStation()
    env.reset()

    teleop_agent = GelloAgent(dynamixel_config, "/dev/ttyUSB1")

    policy = make_lerobot_policy(checkpoint_path, dataset_path)
    lerobot_agent = LerobotAgent(policy, "cuda", preprocessor)

    # create a dataset recorder

    dataset_name = "eval_diffusion_pick_cube"

    if os.path.exists(f"datasets/{dataset_name}"):
        os.system(f"rm -rf datasets/{dataset_name}")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((10,), dtype=np.float32),
        root_dataset_dir=f"datasets/{dataset_name}",
        dataset_name=dataset_name,
        fps=10,
        use_videos=True,
    )

    train_dataset = LeRobotDataset(repo_id="", root=dataset_path)
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    eval(
        env,
        teleop_agent,
        lerobot_agent,
        dataset_recorder,
        policy_to_pose_converter=policy_action_to_abs_se3_converter,
        teleop_to_pose_converter=convert_abs_gello_actions_to_se3,
        fps=10,
        eval_dataset=train_dataset,
        eval_dataset_image_key="observation.images.scene_image",
        env_observation_image_key="scene_image",
    )
