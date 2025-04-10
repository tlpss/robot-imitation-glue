from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.lerobot_dataset.replay_episode import replay_episode
from robot_imitation_glue.ur5station.data_collection import policy_action_to_abs_se3_converter
from robot_imitation_glue.ur5station.ur5_robot_env import UR5eStation

if __name__ == "__main__":
    try:
        env = UR5eStation()
        dataset = LeRobotDataset(repo_id="", root="datasets/pick-cube-remapped")
        replay_episode(
            env, dataset, policy_action_to_abs_se3_converter, "scene_image", "observation.images.scene_image", 0
        )

    finally:
        env.close()
