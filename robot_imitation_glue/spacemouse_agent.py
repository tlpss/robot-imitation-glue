import threading
import time
from collections import deque

import loguru
import pyspacemouse

from robot_imitation_glue.base import BaseAgent

logger = loguru.logger


class SpaceMouseAgent(BaseAgent):
    """spacemouse teleop.
    provides a delta to the current pose of the robot and gripper.
    [x,y,z,rx,ry,rz,gripper]
    """

    ACTION_SPEC = None

    def __init__(self, buffer_size=10):  # Add buffer_size as a parameter
        self.state_buffer = deque(maxlen=buffer_size)  # Use deque as a rolling buffer
        self.running = True

        # separate thread needed for continuous reading of SpaceMouse
        # cf. https://github.com/wuphilipp/gello_software/blob/main/gello/agents/spacemouse_agent.py
        #
        self.thread = threading.Thread(target=self._spacemouse_thread)
        self.thread.daemon = True
        self.thread.start()

    def _spacemouse_thread(self):
        try:
            pyspacemouse.open()
            print("SpaceMouse connected successfully!")
        except Exception as e:
            raise ValueError(f"Could not open SpaceMouse: {e}")

        while self.running:
            try:
                state = pyspacemouse.read()
                if state is not None:
                    self.state_buffer.append(state)
            except Exception as e:
                ValueError(f"Error reading SpaceMouse: {e}")
                break

    def get_action(self, observation=None):
        del observation

        if not self.state_buffer:  # check if buffer is empty.
            logger.warning("SpaceMouse buffer is empty.")
            return [0, 0, 0, 0, 0, 0, 0]

        state = self.state_buffer.pop()

        # do a conversion from the spacemouse coordinate frame to a different coordinate frame that makes teleop more intuitive
        # for the orientations. each twist on the axis corresponds now to how you want the robot to twist as well.
        roll, pitch, yaw = -state.pitch, state.roll, -state.yaw
        rot = [roll, pitch, yaw]

        pos = [state.x, state.y, state.z]

        # TODO: make these configurable
        deadzone_value = 0.1
        rescale_pos = 0.05
        rescale_rot = 0.05

        for i in range(3):
            if abs(pos[i]) < deadzone_value:
                pos[i] = 0
            else:
                pos[i] *= rescale_pos

        for i in range(3):
            if abs(rot[i]) < deadzone_value:
                rot[i] = 0
            else:
                rot[i] *= rescale_rot

        gripper_action = 0
        if state.buttons[1]:
            gripper_action = -0.01
        elif state.buttons[0]:
            gripper_action = 0.01

        return [*pos, *rot, gripper_action]

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


if __name__ == "__main__":
    try:
        agent = SpaceMouseAgent(buffer_size=5)  # set buffer size.
        while True:
            action = agent.get_action()
            print(f"Action: {action}")
            time.sleep(0.05)
    except ValueError as e:
        print(e)
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        if "agent" in locals():
            agent.close()
