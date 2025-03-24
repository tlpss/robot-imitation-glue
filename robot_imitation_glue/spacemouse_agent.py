from robot_imitation_glue.base import BaseAgent

class SpaceMouseAgent(BaseAgent):
    ACTION_SPEC = None

    def __init__(self):
        pass 

    def get_action(self, observation):
        del observation

        # read from space mouse

        # return action as a rotvec that contains the relative