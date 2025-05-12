import json

import click

from robot_imitation_glue.lerobot_dataset.transform_dataset import transform_dataset


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

    def remap_features(frame):
        for old_key, new_key in feature_mapping_dict.items():
            if old_key in frame:
                frame[new_key] = frame.pop(old_key)
        return frame

    transform_dataset(
        repo_id=repo_id,
        new_repo_id=new_repo_id,
        root_dir=root_dir,
        new_root_dir=new_root_dir,
        features_to_drop=features_to_drop_list,
        transform_fn=remap_features,
        transform_features_fn=remap_features,
    )

    click.echo(f"Dataset remapping complete. New dataset saved at {new_root_dir}")


if __name__ == "__main__":
    remap_lerobot_dataset()
