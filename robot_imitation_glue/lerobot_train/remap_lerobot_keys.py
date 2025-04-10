import json
import os
import shutil

import click
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@click.command()
@click.option("--root-dir", required=True, help="Path to the original LeRobot dataset")
@click.option("--new-root-dir", required=True, help="Path to save the remapped LeRobot dataset")
@click.option("--repo-id", default="tlips/lerobot_dataset", help="Repository ID for the original dataset")
@click.option(
    "--new-repo-id",
    default=None,
    help="Repository ID for the new dataset (defaults to original repo_id + '_remapped')",
)
@click.option(
    "--feature-mapping",
    default='{"scene_image": "observation.images.scene_image", "wrist_image": "observation.images.wrist_image", "state": "observation.state"}',
    help="JSON string mapping original feature names to new feature names",
)
@click.option(
    "--features-to-drop",
    default='["wrist_image_original", "scene_image_original"]',
    help="JSON string of feature names to drop from the new dataset",
)
def remap_lerobot_dataset(  # noqa: C901
    root_dir,
    new_root_dir,
    repo_id,
    new_repo_id,
    feature_mapping,
    features_to_drop,
):
    """Remap feature names in a LeRobot dataset.

    This script creates a new LeRobot dataset with renamed features based on the provided mapping.
    The original dataset remains unchanged.
    """
    # Parse JSON strings
    feature_mapping_dict = json.loads(feature_mapping)
    features_to_drop_list = json.loads(features_to_drop)

    # Set default new repo ID if not provided
    if new_repo_id is None:
        new_repo_id = f"{repo_id}_remapped"

    click.echo(f"Remapping features from {root_dir} to {new_root_dir}")
    click.echo(f"Feature mapping: {feature_mapping_dict}")
    click.echo(f"Features to drop: {features_to_drop_list}")

    # Remove existing output directory if it exists
    if os.path.exists(new_root_dir):
        click.echo(f"Removing existing directory: {new_root_dir}")
        shutil.rmtree(new_root_dir)

    # Load the original dataset
    click.echo(f"Loading dataset from {root_dir}")
    dataset = LeRobotDataset(repo_id=repo_id, root=root_dir)

    # Get the original features
    old_features = dataset.features

    # Create new features dictionary with renamed features
    new_features = {}
    for key, value in old_features.items():
        new_features[feature_mapping_dict.get(key, key)] = value

    # Remove features that should be dropped
    for key in features_to_drop_list:
        new_features.pop(key, None)

    # Create a new empty dataset
    click.echo(f"Creating new dataset at {new_root_dir}")
    new_dataset = LeRobotDataset.create(
        repo_id=new_repo_id,
        fps=dataset.fps,
        root=new_root_dir,
        features=new_features,
        use_videos=True,
        image_writer_processes=16,
        image_writer_threads=0,
    )

    # Process each episode in the old dataset
    episode_indices = dataset.episode_data_index
    click.echo(f"Processing {len(episode_indices['from'])} episodes")

    for ep_idx in tqdm.tqdm(range(len(episode_indices["from"]))):
        from_idx, to_idx = episode_indices["from"][ep_idx], episode_indices["to"][ep_idx]
        for idx in range(from_idx, to_idx):
            frame = dataset[idx]
            new_frame = {}
            for key, value in frame.items():
                if key in features_to_drop_list:
                    continue
                if "index" in key or key == "timestamp":
                    continue
                if hasattr(value, "shape") and len(value.shape) == 0:
                    value = value.unsqueeze(0)
                if hasattr(value, "shape") and len(value.shape) == 3:
                    value = value.permute(1, 2, 0)
                new_frame[feature_mapping_dict.get(key, key)] = value

            new_dataset.add_frame(new_frame)
        new_dataset.save_episode()

    click.echo(f"Dataset remapping complete. New dataset saved at {new_root_dir}")


if __name__ == "__main__":
    remap_lerobot_dataset()
