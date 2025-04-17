import time

import numpy as np
from loguru import logger
from openpi_client import image_tools, websocket_client_policy

from robot_imitation_glue.base import BaseAgent


class OpenPIAgent(BaseAgent):
    """
    Remote inference agent that uses the OpenPI policy server to get actions.

    this requires

     1) running OpenPI policy server reachable on this machine (https://github.com/tlpss/openpi/blob/main/docs/remote_inference.md)
         note: for remote SSH-machines, also set up a port forwarding: `ssh -L 8000:localhost:8000 <username>@<host>`
     2) that you have manually installed the openpi client in your current python environment.

    """

    def __init__(
        self, default_prompt: str, observation_preprocessor, host="localhost", port=8000, n_action_steps: int = 8
    ):
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.observation_preprocessor = observation_preprocessor
        self.default_prompt = default_prompt
        self.n_action_steps = n_action_steps  # number of actions from each chunk to execute open loop.
        self.action_queue = []

    def get_action(self, observation):
        get_action_time_start = time.time()
        # if the action queue is empty, we need to get a new chunk of actions from the policy server.
        if len(self.action_queue) == 0:
            observation = self.observation_preprocessor(observation)
            # resize all images to 224x224 with padding, as required by the OpenPI policy inference server
            # cf. https://github.com/tlpss/openpi/blob/main/docs/remote_inference.md
            # TODO: is this something we want to do here or should we do it in the observation preprocessor?

            for k, v in observation.items():
                if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] == 3:
                    observation[k] = image_tools.resize_with_pad(v, 224, 224)

            # add default prompt if not provided
            if "prompt" not in observation.keys():
                observation["prompt"] = self.default_prompt

            time_start = time.time()
            actions = self.client.infer(observation)["actions"]
            time_end = time.time()
            logger.info(f"OpenPI remote inference took {((time_end - time_start)*1000):.2f} ms")
            for i in range(self.n_action_steps):
                self.action_queue.append(actions[i].astype(np.float32))

        action = self.action_queue.pop(0)
        get_action_time_end = time.time()
        logger.info(f"OpenPI agent took {((get_action_time_end - get_action_time_start)*1000):.2f} ms")
        return action


if __name__ == "__main__":
    # Outside of episode loop, initialize the policy client.
    # Point to the host and port of the policy server (localhost and 8000 are the defaults).
    agent = OpenPIAgent(
        default_prompt="", observation_preprocessor=lambda x: x, host="localhost", port=8000, n_action_steps=8
    )
    obs = {
        "state": np.zeros(7),
        "scene_image": np.zeros((240, 320, 3), dtype=np.uint8),
        "wrist_image": np.zeros((240, 320, 3), dtype=np.uint8),
    }

    x = agent.get_action(obs)
    print(x)
    for _ in range(20):
        time_start = time.time()
        x = agent.get_action(obs)
        time_end = time.time()
