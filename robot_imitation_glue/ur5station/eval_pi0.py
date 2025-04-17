import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.agents.gello.gello_agent import GelloAgent
from robot_imitation_glue.agents.openpi_agent import OpenPIAgent
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.ur5station.ur5_robot_env import (
    GELLO_AGENT_PORT,
    UR5eStation,
    abs_joint_policy_action_to_se3,
    convert_abs_gello_actions_to_se3,
    dynamixel_config,
)

if __name__ == "__main__":
    dataset_path = "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-eval-scenarios"

    eval_dataset_name = "pick-cube-eval-pi0"

    def preprocessor(obs_dict):
        scene_img = obs_dict["scene_image"]
        wrist_img = obs_dict["wrist_image"]
        joints = obs_dict["joints"]
        gripper = obs_dict["gripper_state"]
        state = np.concatenate([joints, gripper])

        return {
            "scene_image": scene_img,
            "wrist_image": wrist_img,
            "state": state,
        }

    env = UR5eStation()
    env.reset()

    teleop_agent = GelloAgent(dynamixel_config, GELLO_AGENT_PORT)

    pi0_agent = OpenPIAgent(
        default_prompt="pick up the red cube and place it on the blue square",
        observation_preprocessor=preprocessor,
        host="localhost",
        port=8000,
        n_action_steps=8,
    )

    # create a dataset recorder

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir=f"datasets/{eval_dataset_name}",
        dataset_name=eval_dataset_name,
        fps=10,
        use_videos=True,
    )

    train_dataset = LeRobotDataset(repo_id="", root=dataset_path)
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    eval(
        env,
        teleop_agent,
        pi0_agent,
        dataset_recorder,
        policy_to_pose_converter=abs_joint_policy_action_to_se3,
        teleop_to_pose_converter=convert_abs_gello_actions_to_se3,
        fps=10,
        eval_dataset=train_dataset,
        eval_dataset_image_key="scene_image",
        env_observation_image_key="scene_image",
    )
