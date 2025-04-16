import numpy as np
from ur_analytic_ik import ur5e

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset
from robot_imitation_glue.ur5station.data_collection import policy_action_to_abs_se3_converter

# scenarios:

# 1. convert to joint actions and joint configuration state with absolute gripper


features_to_drop = ["wrist_image_original", "scene_image_original"]


def features_transform(features):
    features["observation.state"] = features.pop("state")
    features["observation.state"]["shape"] = (7,)
    features["observation.images.wrist_image"] = features.pop("wrist_image")
    features["observation.images.scene_image"] = features.pop("scene_image")
    features["action"]["shape"] = (7,)

    print("processed features:")
    print(features)
    return features


def frame_transform(frame):
    current_joints = frame["joints"].numpy()
    action_rot6d = frame["action"].numpy()
    scene_image = frame["scene_image"]
    wrist_image = frame["wrist_image"]

    # convert the action into an SE3 pose
    robot_se3_action, gripper_action = policy_action_to_abs_se3_converter(None, None, action_rot6d)
    gripper_action = np.array([gripper_action])
    # get the pose without TCP transform
    tcp_transform = np.eye(4)
    tcp_transform[2, 3] = 0.184

    robot_joint_action = ur5e.inverse_kinematics_closest_with_tcp(robot_se3_action, tcp_transform, *current_joints)[0]
    robot_joint_action[0] -= 2 * np.pi  # dirty hack to fix a bug in ur analytic IK
    robot_joint_action = np.array(robot_joint_action)
    new_frame = frame.copy()
    new_frame.pop("scene_image")
    new_frame.pop("wrist_image")
    new_frame.pop("state")
    new_frame["observation.state"] = np.concatenate([current_joints, gripper_action]).astype(np.float32)
    new_frame["action"] = np.concatenate([robot_joint_action, gripper_action]).astype(np.float32)
    new_frame["observation.images.scene_image"] = scene_image
    new_frame["observation.images.wrist_image"] = wrist_image

    return new_frame


transform_dataset(
    root_dir="datasets/pick-cube-v2",
    new_root_dir="datasets/pick_cube-v2-remapped-joints",
    transform_fn=frame_transform,
    transform_features_fn=features_transform,
    features_to_drop=features_to_drop,
)
