#!/usr/bin/env python

import argparse
import logging
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def upload_dataset(
    repo_id: str,
    root_dir: str | Path | None = None,
) -> None:
    """Upload a LeRobot dataset to the Hugging Face Hub.

    Args:
        repo_id: Repository ID for the dataset (e.g. 'username/dataset-name').
        root_dir: Path to the local LeRobot dataset. If None, the dataset will be loaded from the hub.
    """
    # Load the dataset
    dataset = LeRobotDataset(repo_id=repo_id, root=root_dir)

    # Upload the dataset to the hub
    dataset.push_to_hub()

    logging.info(f"Successfully uploaded dataset to {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload a LeRobot dataset to the Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID for the dataset (e.g. 'username/dataset-name').",
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Path to the local LeRobot dataset. If not provided, the dataset will be loaded from the hub.",
    )

    args = parser.parse_args()

    upload_dataset(
        repo_id=args.repo_id,
        root_dir=args.root_dir,
    )


if __name__ == "__main__":
    main()
