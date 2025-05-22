import numpy as np
from airo_spatial_algebra import SE3Container
from scipy.spatial.transform import Rotation as R


def delta_action_to_abs_se3_converter(robot_pose_se3, gripper_state, action):
    # convert spacemouse action to ur3e action
    # we take the action to consist of a delta position, delta rotation and delta gripper width.
    # the delta rotation is interpreted as expressed in a frame with the same orientation as the base frame but with the origin at the EEF.
    # in this way, when rotating the spacemouse, the robot eef will not move around, while at the same time the axes of orientation
    # do not depend on the current orientation of the EEF.

    # the delta position is intepreted in the world frame and also applied on the EEF frame.

    delta_pos = action[:3]
    delta_rot = action[3:6]
    gripper_action = action[6]

    robot_trans = robot_pose_se3[:3, 3]
    robot_SO3 = robot_pose_se3[:3, :3]

    new_robot_trans = robot_trans + delta_pos
    # rotation is now interpreted as euler and not as rotvec
    # similar to Diffusion Policy.
    # however, rotvec seems more principled (related to twist)
    new_robot_SO3 = R.from_euler("xyz", delta_rot).as_matrix() @ robot_SO3

    new_robot_SE3 = np.eye(4)
    new_robot_SE3[:3, :3] = new_robot_SO3
    new_robot_SE3[:3, 3] = new_robot_trans

    new_gripper_state = gripper_state + gripper_action
    new_gripper_state = np.clip(new_gripper_state, 0, 0.085)

    return new_robot_SE3, new_gripper_state


def convert_policy(robot_pose_se3, gripper_state, action):
    gripper_action = action[6]
    relative_robot_pose = SE3Container.from_euler_angles_and_translation(action[3:6], action[0:3]).homogeneous_matrix
    target_pose_se3 = robot_pose_se3 @ relative_robot_pose

    new_gripper_state = 0 if gripper_action < 0.5 else 0.085
    return target_pose_se3, new_gripper_state


if __name__ == "__main__":
    """example of how to use the eval function"""
    import os

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.eval_agent import eval
    from robot_imitation_glue.openvla_agent import OpenVLAAgent
    from robot_imitation_glue.robot_env import UR3eStation
    from robot_imitation_glue.spacemouse_agent import SpaceMouseAgent

    env = UR3eStation()
    env.reset()
    teleop_agent = SpaceMouseAgent()
    policy_agent = OpenVLAAgent()

    if os.path.exists("datasets/eval_dataset"):
        dataset = LeRobotDataset(repo_id="", root="datasets/eval_dataset")
    else:
        dataset = None
    # create a dataset recorder

    if os.path.exists("datasets/eval_openvla_green"):
        os.system("rm -rf datasets/eval_openvla_green")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir="datasets/eval_openvla_green",
        dataset_name="eval_openvla_green",
        fps=10,
        use_videos=True,
    )

    eval(
        env,
        teleop_agent,
        policy_agent,
        dataset_recorder,
        convert_policy,
        delta_action_to_abs_se3_converter,
        fps=10,
        eval_dataset=dataset,
        eval_dataset_image_key="scene_image",
    )
