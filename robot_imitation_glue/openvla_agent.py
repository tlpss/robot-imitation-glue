import json_numpy
import requests

from robot_imitation_glue.base import BaseAgent

json_numpy.patch()
import math

import numpy as np
from PIL import Image


class OpenVLAAgent(BaseAgent):
    """OpenVLA policy agent"""

    ACTION_SPEC = None

    def __init__(self):
        self.counter = 0

    def reset(self):
        # Do nothing, is only for agents with action chunking
        pass

    def get_action(self, observation=None):
        # Get observation image
        image = observation["scene_image"]
        image = self.augment(image)

        # Language instruction & unnorm key are currently hardcoded
        instruction = "move the block to the green circle"
        unnorm_key = "lerobot_rlds"

        action = requests.post(
            "http://0.0.0.0:8000/act", json={"image": image, "instruction": instruction, "unnorm_key": unnorm_key}
        ).json()
        action = action.astype(np.float32)
        print(action)
        return action

    def center_crop(self, img, new_width=None, new_height=None):

        width = img.shape[1]
        height = img.shape[0]

        if new_width is None:
            new_width = min(width, height)

        if new_height is None:
            new_height = min(width, height)

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 2))
        bottom = height - int(np.floor((height - new_height) / 2))

        if len(img.shape) == 2:
            center_cropped_img = img[top:bottom, left:right]
        else:
            center_cropped_img = img[top:bottom, left:right, ...]

        return center_cropped_img

    def augment(self, temp_image):
        crop_scale = 0.9
        sqrt_crop_scale = math.sqrt(crop_scale)
        temp_image_cropped = self.center_crop(
            temp_image, int(sqrt_crop_scale * temp_image.shape[1]), int(sqrt_crop_scale * temp_image.shape[0])
        )
        temp_image = Image.fromarray(temp_image_cropped)
        temp_image = temp_image.resize((224, 224), Image.Resampling.BILINEAR)  # IMPORTANT: dlimp uses BILINEAR resize
        image = temp_image

        # Convert to numpy array
        image = np.array(image)

        return image
