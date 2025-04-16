import os
import shutil
from typing import Callable, Dict, Optional

import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def transform_dataset(  # noqa: C901
    repo_id: Optional[str] = None,
    root_dir: Optional[str] = None,
    new_root_dir: str = None,
    new_repo_id: Optional[str] = None,
    transform_fn: Callable[[Dict], Dict] = None,
    transform_features_fn: Optional[Callable[[Dict], Dict]] = None,
    features_to_drop: Optional[list] = None,
    use_videos: bool = True,
    image_writer_processes: int = 0,
    image_writer_threads: int = 16,
    verbose: bool = True,
) -> LeRobotDataset:
    """Transform a LeRobot dataset using custom mapping functions.

    This function creates a new LeRobot dataset by applying transformation functions
    to both the dataset features and each frame in the original dataset.
    The original dataset remains unchanged.

    Args:
        repo_id: Repository ID for the original dataset. Either repo_id or root_dir must be provided.
        root_dir: Path to the original LeRobot dataset. Either repo_id or root_dir must be provided.
        new_root_dir: Path to save the transformed LeRobot dataset.
        new_repo_id: Repository ID for the new dataset (defaults to original repo_id + '_transformed').
        transform_fn: Function that takes a frame dictionary and returns a transformed frame dictionary.
        transform_features_fn: Optional function that takes a features dictionary and returns a transformed features dictionary.
        features_to_drop: List of feature names to drop from the new dataset.
        use_videos: Whether to use videos for the new dataset.
        image_writer_processes: Number of processes for image writing.
        image_writer_threads: Number of threads for image writing.
        verbose: Whether to print progress information.

    Returns:
        The transformed LeRobot dataset.

    Raises:
        ValueError: If neither repo_id nor root_dir is provided.
    """
    if repo_id is None and root_dir is None:
        raise ValueError("Either repo_id or root_dir must be provided.")

    if transform_fn is None:
        raise ValueError("transform_fn must be provided.")

    # Set default new repo ID if not provided
    if new_repo_id is None and repo_id is not None:
        new_repo_id = f"{repo_id}_transformed"

    if verbose:
        print(f"Transforming dataset from {root_dir or repo_id} to {new_root_dir}")
        if features_to_drop:
            print(f"Features to drop: {features_to_drop}")

    # Remove existing output directory if it exists
    if os.path.exists(new_root_dir):
        if verbose:
            print(f"Removing existing directory: {new_root_dir}")
        shutil.rmtree(new_root_dir)

    # Load the original dataset
    if verbose:
        print(f"Loading dataset from {root_dir or repo_id}")

    dataset = LeRobotDataset(repo_id=repo_id, root=root_dir)

    # Get the original features
    old_features = dataset.features

    # Create new features dictionary
    new_features = {}
    for key, value in old_features.items():
        if features_to_drop and key in features_to_drop:
            continue
        new_features[key] = value

    # Apply feature transformation if provided
    if transform_features_fn is not None:
        new_features = transform_features_fn(new_features)

    # Create a new empty dataset
    if verbose:
        print(f"Creating new dataset at {new_root_dir}")

    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=dataset.fps,
        root=new_root_dir,
        features=new_features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )

    # Process each episode in the old dataset
    episode_indices = dataset.episode_data_index
    if verbose:
        print(f"Processing {len(episode_indices['from'])} episodes")

    for ep_idx in tqdm.tqdm(range(len(episode_indices["from"])), disable=not verbose):
        from_idx, to_idx = episode_indices["from"][ep_idx], episode_indices["to"][ep_idx]
        for idx in range(from_idx, to_idx):
            frame = dataset[idx]

            # these are auto-generated, so remove them from the frame
            features_to_drop_list = ["index", "timestamp", "frame_index", "episode_index", "task_index"]
            if features_to_drop:
                features_to_drop_list.extend(features_to_drop)
            frame = {k: v for k, v in frame.items() if k not in features_to_drop_list}

            for key, value in frame.items():
                # drop auto-generated meta data
                # convert 0D tensors to 1D tensors
                if hasattr(value, "shape") and len(value.shape) == 0:
                    value = value.unsqueeze(0)
                    frame[key] = value
                # make images channel-last again
                if hasattr(value, "shape") and len(value.shape) == 3:
                    value = value.permute(1, 2, 0)
                    frame[key] = value

            # Apply the transformation function
            new_frame = transform_fn(frame)

            # Add the transformed frame to the new dataset
            new_dataset.add_frame(new_frame)

        new_dataset.save_episode()

    if verbose:
        print(f"Dataset transformation complete. New dataset saved at {new_root_dir}")

    return new_dataset


# Example usage:
if __name__ == "__main__":  # noqa: C901
    # Example transformation function that renames features
    def example_transform(frame):
        new_frame = {}

        for key, value in frame.items():
            # Example mapping
            if key == "scene_image":
                new_frame["observation.images.scene_image"] = value
            elif key == "wrist_image":
                new_frame["observation.images.wrist_image"] = value
            elif key == "joints":
                new_frame["observation.joints"] = value
            elif key == "gripper_state":
                new_frame["observation.gripper_state"] = value
            else:
                new_frame[key] = value

        return new_frame

    # Example feature transformation function
    def example_transform_features(features):
        new_features = {}
        for key, value in features.items():
            # Example mapping for features
            if key == "scene_image":
                new_features["observation.images.scene_image"] = value
            elif key == "wrist_image":
                new_features["observation.images.wrist_image"] = value
            elif key == "joints":
                new_features["observation.joints"] = value
            elif key == "gripper_state":
                new_features["observation.gripper_state"] = value
            else:
                new_features[key] = value
        return new_features

    # Example call
    transform_dataset(
        root_dir="datasets/pick-cube-val",
        new_root_dir="datasets/pick_cube-transformed-test",
        transform_fn=example_transform,
        transform_features_fn=example_transform_features,
        features_to_drop=["wrist_image_original", "scene_image_original"],
    )
