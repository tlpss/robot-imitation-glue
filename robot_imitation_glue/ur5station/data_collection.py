from datetime import datetime
from pathlib import Path

import numpy as np

from robot_imitation_glue.agents.gello import DynamixelConfig, GelloAgent
from robot_imitation_glue.collect_data import collect_data
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation, convert_abs_gello_actions_to_se3


def abs_se3_to_policy_action_converter(robot_pose, gripper_pose, abs_se3_action, gripper_action):
    """absolute poses, encoded as position, x-vector of rotation, y-vector of rotation, gripper action"""
    abs_position_target = abs_se3_action[:3, 3]
    abs_rotation_x = abs_se3_action[:3, 0]
    abs_rotation_y = abs_se3_action[:3, 1]

    policy_action = np.zeros(10)
    policy_action[:3] = abs_position_target
    policy_action[3:6] = abs_rotation_x
    policy_action[6:9] = abs_rotation_y
    policy_action[9] = gripper_action
    policy_action = policy_action.astype(np.float32)
    return policy_action


def policy_action_to_abs_se3_converter(robot_pose, gripper_pose, policy_action):
    abs_position_target = policy_action[:3]
    abs_rotation_x = policy_action[3:6]
    abs_rotation_y = policy_action[6:9]
    gripper_action = policy_action[9]

    # Gramm-schmidt orthonormalization
    abs_rotation_x = abs_rotation_x / np.linalg.norm(abs_rotation_x)
    abs_rotation_y = abs_rotation_y / np.linalg.norm(abs_rotation_y)
    abs_rotation_z = np.cross(abs_rotation_x, abs_rotation_y)
    abs_rotation_z = abs_rotation_z / np.linalg.norm(abs_rotation_z)
    abs_rotation_y = np.cross(abs_rotation_z, abs_rotation_x)
    abs_rotation_y = abs_rotation_y / np.linalg.norm(abs_rotation_y)

    # convert to se3
    target_pose = np.eye(4)
    target_pose[:3, 3] = abs_position_target
    target_pose[:3, 0] = abs_rotation_x
    target_pose[:3, 1] = abs_rotation_y
    target_pose[:3, 2] = abs_rotation_z
    return target_pose, gripper_action


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    args = parser.parse_args()

    env = UR5eStation()

    config = DynamixelConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],
        joint_offsets=(np.array([40, 16, 25, 40, 15, 7]) * np.pi / 16).tolist(),
        joint_signs=[1, 1, -1, 1, 1, 1],
        gripper_config=(7, 194, 152),
    )
    agent = GelloAgent(config, "/dev/ttyUSB1")

    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros(10),
        root_dataset_dir=Path(f"datasets/{args.dataset_name}/"),
        dataset_name="none",
        fps=10,
    )

    input("Press Enter to start collecting data")
    action = agent.get_action(env.get_observations())
    initial_pose, gripper = convert_abs_gello_actions_to_se3(
        env.get_robot_pose_se3(), env.get_gripper_opening(), action
    )
    # first move robot slowly to the current teleop pose.
    env.robot.move_linear_to_tcp_pose(initial_pose).wait()

    collect_data(
        env,
        agent,
        dataset_recorder,
        frequency=10,
        teleop_to_pose_converter=convert_abs_gello_actions_to_se3,
        abs_pose_to_policy_action=abs_se3_to_policy_action_converter,
    )
