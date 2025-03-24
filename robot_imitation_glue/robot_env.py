from robot_imitation_glue.base import BaseEnv

class UR3eEnv(BaseEnv):
    ACTION_SPEC = None
    PROPRIO_OBS_SPEC = None

    def __init__(self):
        
        # set up UR3e
        
        # set up cameras

        # set up additional sensors if needed.
        pass
    
    def get_observations():
        obs_dict = {}
        return obs_dict

    def act(action):
        pass