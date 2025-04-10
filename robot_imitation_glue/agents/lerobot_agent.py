import time

import numpy as np
import torch
from loguru import logger

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.diffusion.configuration_diffusion import PreTrainedConfig
from lerobot.common.policies.factory import make_policy
from robot_imitation_glue.base import BaseAgent


def make_lerobot_policy(pretrained_path, dataset_path):
    """ """
    # TODO: try to omit the need to load the dataset, bc it is not always on the same machine and is a source of errors..
    # not sure why Lerobot has not simply stored the metadata in an additional file.
    # config["pretrained_path"] = path
    # rmeove "type" key from policy_config
    policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
    dataset = LeRobotDataset(repo_id="dataset", root=dataset_path)
    policy = make_policy(policy_config, ds_meta=dataset.meta)
    policy.eval()
    return policy


class LerobotAgent(BaseAgent):
    """
    agent for inference on a policy trained with Lerobot.

    """

    def __init__(self, policy, device, observation_preprocessor):
        """
        processor must take the env obs dict and do
        1) numpy to tensor
        2) batchifying the observation
        3) renaming the keys to the policy expected keys
        4) (optional) do any other preprocessing, such as image resizing/cropping...


        """
        super().__init__()
        self.policy = policy
        self.device = device
        self.observation_preprocessor = observation_preprocessor

    def get_action(self, observation):
        start_time = time.time()
        observation = self.observation_preprocessor(observation)
        end_time = time.time()
        logger.info(f"Lerobot agent observation preprocessor took {((end_time - start_time)*1000):.2f} ms")
        observation = {k: v.to(self.device) for k, v in observation.items()}
        with torch.no_grad():
            time_start = time.time()
            action = self.policy.select_action(observation)
            time_end = time.time()
            logger.info(f"Lerobot agent inference took {((time_end - time_start)*1000):.2f} ms")
        return action.squeeze(0).cpu().numpy()


if __name__ == "__main__":

    path = "/home/tlips/Code/robot-imitation-glue/outputs/train/2025-04-10/13-15-24_pick-cube_diffusion/checkpoints/035000/pretrained_model"
    dataset_path = "/home/tlips/Code/robot-imitation-glue/datasets/pick-cube-remapped"

    policy = make_lerobot_policy(path, dataset_path).cpu()

    dataset = LeRobotDataset(repo_id="dataset", root=dataset_path)

    # test policy

    batch = dataset[0]
    action = policy.select_action(batch)
    print(f"policy action: {action}")
    print(f"dataset ['action']: {batch['action']}")

    print("testing inference on dummy observations")

    def observation_preprocessor(observation):
        observation["img1"] = observation["img1"].transpose(2, 0, 1)
        observation["img2"] = observation["img2"].transpose(2, 0, 1)
        observation["img1"] = observation["img1"].astype(np.float32) / 255.0
        observation["img2"] = observation["img2"].astype(np.float32) / 255.0

        observation["observation.images.scene_image"] = torch.from_numpy(observation["img1"]).float()
        observation["observation.images.wrist_image"] = torch.from_numpy(observation["img2"]).float()
        observation["observation.state"] = torch.from_numpy(observation["state"]).float()

        # drop old keys
        observation.pop("img1")
        observation.pop("img2")
        observation.pop("state")

        # resize images to 224x224
        from torchvision import transforms

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(196)])
        observation["observation.images.scene_image"] = transform(observation["observation.images.scene_image"])
        observation["observation.images.wrist_image"] = transform(observation["observation.images.wrist_image"])

        # add batch dimension
        observation["observation.images.scene_image"] = observation["observation.images.scene_image"].unsqueeze(0)
        observation["observation.images.wrist_image"] = observation["observation.images.wrist_image"].unsqueeze(0)
        observation["observation.state"] = observation["observation.state"].unsqueeze(0)
        return observation

    policy = make_lerobot_policy(path, dataset_path)
    policy = policy.to("cuda")
    agent = LerobotAgent(policy, "cuda", observation_preprocessor)
    for _ in range(20):
        test_obs = {
            "img1": np.random.randint(0, 255, (256, 256, 3)),
            "img2": np.random.randint(0, 255, (256, 256, 3)),
            "state": np.random.randn(7),
        }
        action = agent.get_action(test_obs)
    print(f"agent action: {action}")
