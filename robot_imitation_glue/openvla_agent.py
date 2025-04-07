import json_numpy
import requests

from robot_imitation_glue.base import BaseAgent

json_numpy.patch()
import numpy as np


class OpenVLAAgent(BaseAgent):
    """OpenVLA policy agent"""

    ACTION_SPEC = None

    def __init__(self):
        pass

    def get_action(self, observation=None):
        # Get observation image
        image = observation["scene_image"]

        # Get language instruction, currently hardcoded
        instruction = "move the block to the blue target"

        # Note: dataset name is hardcoded for now
        action = requests.post(
            "http://0.0.0.0:8000/act", json={"image": image, "instruction": instruction, "unnorm_key": "lerobot_rlds"}
        ).json()
        action = action.astype(np.float32)
        print(action)
        return action
