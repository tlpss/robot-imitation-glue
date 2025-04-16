from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.lerobot_dataset.replay_episode import replay_episode
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation, abs_joint_policy_action_to_se3

if __name__ == "__main__":
    try:
        env = UR5eStation()
        dataset = LeRobotDataset(repo_id="", root="datasets/pick_cube-transformed-test")
        replay_episode(
            env, dataset, abs_joint_policy_action_to_se3, "scene_image", "observation.images.scene_image", 0
        )

    finally:
        env.close()
