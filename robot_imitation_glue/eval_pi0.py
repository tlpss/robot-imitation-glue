import numpy as np
from scipy.spatial.transform import Rotation as R
from ur_analytic_ik import ur3e

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
from robot_imitation_glue.eval_agent import eval
from robot_imitation_glue.openpi_agent import OpenPIAgent
from robot_imitation_glue.robot_env import UR3eStation
from robot_imitation_glue.spacemouse_agent import SpaceMouseAgent

if __name__ == "__main__":
    dataset_path = "datasets/init_cond_task2"

    eval_dataset_name = "eval_pi0_task2_objvar"

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

    env = UR3eStation()
    env.reset()

    teleop_agent = SpaceMouseAgent()

    pi0_agent = OpenPIAgent(
        default_prompt="attach the circuit breaker to the DIN rail",
        observation_preprocessor=preprocessor,
        host="localhost",
        port=8000,
        n_action_steps=15,
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

    def abs_joint_policy_action_to_se3(current_pose, current_gripper_state, action: np.ndarray):
        del current_pose, current_gripper_state
        SCHUNK_TCP_OFFSET = 0.184
        joints = action[:6]
        gripper = action[6]
        tcp_pose = np.eye(4)
        tcp_pose[2, 3] = SCHUNK_TCP_OFFSET
        pose = ur3e.forward_kinematics_with_tcp(*joints, tcp_pose)
        return pose, gripper

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

    train_dataset = LeRobotDataset(repo_id="", root=dataset_path)
    input("Press Enter to start evaluation (should hold your teleop in place now!)")
    eval(
        env,
        teleop_agent,
        pi0_agent,
        dataset_recorder,
        policy_to_pose_converter=abs_joint_policy_action_to_se3,
        teleop_to_pose_converter=delta_action_to_abs_se3_converter,
        fps=10,
        eval_dataset=train_dataset,
        eval_dataset_image_key="scene_image",
        # env_observation_image_key="scene_image",
    )
