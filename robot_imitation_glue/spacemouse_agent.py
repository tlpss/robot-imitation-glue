import pyspacemouse
from scipy.spatial.transform import Rotation as R

from robot_imitation_glue.base import BaseAgent


class SpaceMouseAgent(BaseAgent):
    ACTION_SPEC = None

    def __init__(self):
        # Open spacemouse and verify that it is found
        mouse_found = pyspacemouse.open()
        assert mouse_found, "No SpaceMouse found, exiting"

        # self.gripper_state = 0 # Has to be read from the gripper and given to the agent

    def get_action(self, observation):
        del observation

        # read from space mouse
        state = pyspacemouse.read()

        # return relative action as a rotvec
        roll, pitch, yaw = state.roll, state.pitch, state.yaw
        rotvec = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_rotvec()

        # TODO gripper action: absolute (how? with gripper state inside this agent?) or relative?
        gripper_action = 0

        # if state.buttons[1]: # If the right button is pressed (or both)
        #     gripper_action = 0 # Closed gripper
        # elif state.buttons[0]:  # If only the left button is pressed (or both)
        #     gripper_action = 1 # Open gripper
        # else: # If no button is pressed
        #     gripper_action = self.gripper_state # Keep the current gripper state

        return [state.x, state.y, state.z, *rotvec, gripper_action]


# Test the agent
if __name__ == "__main__":
    agent = SpaceMouseAgent()
    while True:
        print(agent.get_action(None))
