
from pathlib import Path
import torch
import numpy as np 
from robot_imitation_glue.base import BaseDatasetRecorder


class DummyDatasetRecorder(BaseDatasetRecorder):
    def start_episode(self):
        print("starting dataset episode recording")

    def record_step(self, obs, action):
        print("recording step")
        print("saving obs:", obs)
        print("saving action:", action)

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

    def __init__(self, example_obs_dict: dict, example_action: np.array, root_dataset_dir: Path, dataset_name: str, fps: int, use_videos=True):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.root_dataset_dir = root_dataset_dir
        self.dataset_name = dataset_name
        self.fps = fps

        self._n_recorded_episodes = 0
        self.key_mapping_dict = {}


        self.image_keys = []
        self.state_keys = []

        # create features  using the example dict. assume all numpy arrays. 0D is a scalar, 1D with size 1 is a scalar, 1D with size > 1 is a vector, 3D is an image.
        features = self.DEFAULT_FEATURES.copy()
        for key, value in example_obs_dict.items():
            shape = value.shape
            if len(shape) == 0:
                features[key] = {"dtype": "float32", "shape": (1,), "names": None}
            elif len(shape) == 1:
                features[key] = {"dtype": "float32", "shape": shape, "names": None}
            elif len(shape) == 3:
                if use_videos:
                    features[key] = {"dtype": "video", "names": ["channel", "height", "width"], "shape": shape}
                else:
                    features[key] = {"dtype": "image", "shape": shape, "names": None}
            else:
                raise ValueError(f"Unsupported shape for feature {key}: {shape}")

            if  len(shape) == 3:
                self.image_keys.append(key)
            else:
                self.state_keys.append(key)
        
        # add action to features
        features["action"] = {"dtype": "float32", "shape": example_action.shape, "names": None}
        print(f"Features: {features}")

        #TODO: if dataset exists, load it to extend it.
        
        self.lerobot_dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=self.fps,
            root=self.root_dataset_dir,
            features=features,
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=4,
        )

    def start_episode(self):
        pass

    def record_step(self, obs, action):
        frame = {
            "action": torch.from_numpy(action),
            "next.reward": torch.tensor([0.0]),
            "next.success": torch.tensor([False]),
            "seed": torch.tensor([0]),  # TODO: store the seed
            "task":""
        }
        for key in self.image_keys:
            lerobot_key = self.key_mapping_dict.get(key, key)
            frame[lerobot_key] = obs[key]

        for key in self.state_keys:
            frame[key] = torch.from_numpy(obs[key])
        self.lerobot_dataset.add_frame(frame)

    def save_episode(self):
        self.lerobot_dataset.save_episode()
        self._n_recorded_episodes += 1

    def finish_recording(self):
        pass

    @property
    def n_recorded_episodes(self):
        return self._n_recorded_episodes
    

if __name__ == "__main__":

    example_obs = {
        "robot_pose": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "image": np.random.rand(3, 64, 64).astype(np.float32),
        "image2": np.random.rand(3, 128, 64).astype(np.float32),
        "gripper_state": np.array([0.1],dtype=np.float32),

    }

    example_action = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],dtype=np.float32)

    import os 
    # remove entire dataset directory
    os.system("rm -rf datasets")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=example_obs,
        example_action=example_action,
        root_dataset_dir=Path("datasets"),
        dataset_name="test_dataset",
        fps=30,
        use_videos=False,
    )

    for j in range(3):
        dataset_recorder.start_episode()
        for i in range(10-j):
            dataset_recorder.record_step(example_obs, example_action)
        dataset_recorder.save_episode()

    dataset_recorder.finish_recording()
    print(f"Recorded {dataset_recorder.n_recorded_episodes} episodes.")

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id="test_dataset", root=Path("datasets"),episodes=[0,1])
    print(f"Loaded {len(dataset)} steps.")
    print(dataset.episode_data_index)


