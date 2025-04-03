import numpy as np

from robot_imitation_glue.base import BaseAgent, BaseEnv


class MockEnv(BaseEnv):
    """
    a very simple mock environment where a 2D point is moved around in a grid.

    The environment has a fixed size and the agent can move in 4 directions: up, down, left, right.
    """

    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        self.target_pos = np.array([0, 0], dtype=np.int32)

    def reset(self):
        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2], dtype=np.int32)
        self.target_pos = np.random.randint(0, self.grid_size, size=(2,), dtype=np.int32)

    def get_observations(self):
        state_img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        state_img[self.agent_pos[0], self.agent_pos[1], :] = state_img[
            self.agent_pos[0], self.agent_pos[1], :
        ] + np.array(
            [255, 0, 0]
        )  # red color for the agent
        state_img[self.target_pos[0], self.target_pos[1]] = state_img[
            self.target_pos[0], self.target_pos[1], :
        ] + np.array(
            [
                0,
                0,
                255,
            ]
        )  # blue color for the target

        return {"agent_pos": self.agent_pos.astype(np.float32), "scene": state_img}

    def act(self, robot_pose_se3, gripper_opening, timestamp):
        action = robot_pose_se3[:2, 3]
        # binarize action
        action = np.round(action).astype(np.int32)
        self.agent_pos = action
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)

    def get_robot_pose_se3(self):
        pose = np.eye(4)
        pose[:2, 3] = self.agent_pos
        return pose

    def get_gripper_opening(self):
        return 0.0


class MockAgent(BaseAgent):
    """
    a very simple mock agent that provides 2D translations using keyboard arrows
    """

    def __init__(self):
        super().__init__()
        import pynput

        self.kb_press_action = np.zeros(2)

        def on_press(key):
            if hasattr(key, "char"):
                if key.char == "j":
                    self.kb_press_action[1] = -1
                elif key.char == "l":
                    self.kb_press_action[1] = 1
                elif key.char == "i":
                    self.kb_press_action[0] = -1
                elif key.char == "k":
                    self.kb_press_action[0] = 1

        self.listener = pynput.keyboard.Listener(on_press=on_press)
        self.listener.start()

    def get_action(self, observations):
        del observations
        # get the action from the keyboard

        action = np.zeros((7,), dtype=np.float32)
        action[:2] = self.kb_press_action
        self.kb_press_action = np.zeros(2)
        return action

    def close(self):
        pass


def mock_agent_to_pose_converter(robot_pose, gripper_pose, action):
    # convert the action to a 2D translation
    new_robot_pose = robot_pose.copy()
    new_robot_pose[:2, 3] += action[:2]
    return new_robot_pose, gripper_pose


if __name__ == "__main__":
    import cv2

    env = MockEnv()
    env.reset()
    agent = MockAgent()

    pose = env.get_robot_pose_se3()
    while True:
        import time

        observations = env.get_observations()
        # convert teleop action to env action
        img = observations["scene"]
        cv2.imshow("scene", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        action = agent.get_action(observations)
        env_action, env_gripper = mock_agent_to_pose_converter(pose, env.get_gripper_opening(), action)
        env.act(robot_pose_se3=env_action, gripper_opening=env_gripper, timestamp=0)
        pose = env_action
        time.sleep(0.1)
