
from pathlib import Path
import torch
import numpy as np 
from robot_imitation_glue.base import BaseEnv, BaseDatasetRecorder


class DummyDatasetRecorder(BaseDatasetRecorder):
    def start_episode(self):
        print("starting dataset episode recording")

    def record_step(self, obs, action):
        print("recording step")

    def save_episode(self):
        print("saving dataset episode")


class LeRobotDatasetRecorder(BaseDatasetRecorder):
    DEFAULT_FEATURES = {
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "seed": {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        },
        "timestamp": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
    }

    def __init__(self, env: BaseEnv, root_dataset_dir: Path, dataset_name: str, fps: int, use_videos=True):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.root_dataset_dir = root_dataset_dir
        self.dataset_name = dataset_name
        self.fps = fps

        self._n_recorded_episodes = 0
        self.key_mapping_dict = {}


        # TODO: either add mapping to robot env OR infer from an observation DICT.


        # features = self.DEFAULT_FEATURES.copy()
        # # add images to features

        # # uses the lerobot convention to map to 'observation.image' keys
        # # and stores as video.

        # assert isinstance(env.observation_space, gymnasium.spaces.Dict), "Observation space should be a dict"
        # self.image_keys = [key for key in env.observation_space.spaces.keys() if "image" in key]
        # num_cameras = len(self.image_keys)
        # for key in self.image_keys:
        #     shape = env.observation_space.spaces[key].shape

        #     if not key.startswith("observation.images"):
        #         lerobot_key = f"observation.images.{key}"
        #         self.key_mapping_dict[key] = lerobot_key

        #     lerobot_key = self.key_mapping_dict.get(key, key)
        #     if "/" in lerobot_key:
        #         self.key_mapping_dict[key] = lerobot_key.replace("/", "_")
        #     lerobot_key = self.key_mapping_dict[key]
        #     if use_videos:
        #         features[lerobot_key] = {"dtype": "video", "names": ["channel", "height", "width"], "shape": shape}
        #     else:
        #         features[lerobot_key] = {"dtype": "image", "shape": shape, "names": None}

        # # state observations
        # self.state_keys = [key for key in env.observation_space.spaces.keys() if key not in self.image_keys]
        # for key in self.state_keys:
        #     shape = env.observation_space.spaces[key].shape
        #     features[key] = {"dtype": "float32", "shape": shape, "names": None}

        # # add single 'state' observation that concatenates all state observations
        # features["observation.state"] = {
        #     "dtype": "float32",
        #     "shape": (sum([env.observation_space.spaces[key].shape[0] for key in self.state_keys]),),
        #     "names": None,
        # }
        # # add action to features
        # features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

        print(f"Features: {features}")
        # create the dataset
        self.lerobot_dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=self.fps,
            root=self.root_dataset_dir,
            features=features,
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=4 * num_cameras,
        )

    def start_episode(self):
        pass

    def record_step(self, obs, action):
        timestamp = self.lerobot_dataset.episode_buffer["size"] / self.fps

        frame = {
            "action": torch.from_numpy(action),
            "next.reward": torch.tensor(0.0),
            "next.success": torch.tensor(False),
            "seed": torch.tensor(0),  # TODO: store the seed
            "timestamp": timestamp,
        }
        for key in self.image_keys:
            lerobot_key = self.key_mapping_dict.get(key, key)
            frame[lerobot_key] = obs[key]

        for key in self.state_keys:
            frame[key] = torch.from_numpy(obs[key])

        # concatenate all 'state' observations into a single tensor
        state = torch.cat([frame[key].flatten() for key in self.state_keys])
        frame["observation.state"] = state

        self.lerobot_dataset.add_frame(frame)

    def save_episode(self):
        self.lerobot_dataset.save_episode(task="")
        self._n_recorded_episodes += 1

    def finish_recording(self):
        # computing statistics
        self.lerobot_dataset.consolidate()

    @property
    def n_recorded_episodes(self):
        return self._n_recorded_episodes