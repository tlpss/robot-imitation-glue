import time

import numpy as np
from loguru import logger

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.base import BaseEnv


def replay_episode(
    env: BaseEnv,
    dataset: LeRobotDataset,
    action_to_env_converter,
    image_key,
    dataset_image_key,
    episode_idx: int = 0,
    fps=10,
):
    episode_indices = dataset.episode_data_index
    episode_start_idx = episode_indices["from"][episode_idx].item()
    episode_to_idx = episode_indices["to"][episode_idx].item()

    dataset_initial_image = dataset[episode_start_idx][dataset_image_key]
    dataset_initial_action = dataset[episode_start_idx]["action"].cpu().numpy()
    initial_robot_pose, initial_gripper = action_to_env_converter(
        env.get_robot_pose_se3(), env.get_gripper_opening(), dataset_initial_action
    )

    input(f"Press Enter to move robot to initial pose \n {initial_robot_pose}")
    env.move_robot_to_tcp_pose(initial_robot_pose)
    env.move_gripper(initial_gripper)

    # convert torch image to numpy image
    dataset_initial_image = dataset_initial_image.cpu().numpy()
    dataset_initial_image = dataset_initial_image.transpose(1, 2, 0).astype(np.uint8)

    # while True:
    #     img = env.get_observations()[image_key]

    #     # blend the two images
    #     blended_image = cv2.addWeighted(dataset_initial_image, 0.5, img, 0.5, 0)
    #     print(f"blended_image.shape = {blended_image.shape}")
    #     cv2.imshow("image", blended_image)
    #     print(f"img.shape = {img.shape}")
    #     k = cv2.waitKey(1)
    #     print(f"k = {k}")
    #     if k == ord('q'):
    #         break

    input("Press Enter to start replay")

    duration = 1.0 / fps
    for i in range(episode_start_idx, episode_to_idx):
        action = dataset[i]["action"].cpu().numpy()
        obs = env.get_observations()
        # print(f"current obs = {obs}")
        # print(f"dataset obs = {dataset[i]}")
        robot_pose, gripper = action_to_env_converter(env.get_robot_pose_se3(), env.get_gripper_opening(), action)
        logger.debug(f"target robot pose = {robot_pose}")
        logger.debug(f"current robot pose = {env.get_robot_pose_se3()}")
        logger.debug(f"current state observation = {obs['state']}")
        env.act(robot_pose, gripper, time.time() + duration)
        time.sleep(duration)

    print("replay finished")
