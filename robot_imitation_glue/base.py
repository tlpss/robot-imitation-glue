class BaseEnv:
    def get_observations():
        pass
    def act(action):
        pass
    def reset():
        pass


class BaseAgent:
    def get_action(self, observation):
        pass




# define different action types
# ABS_EEF, REL_EEF, ABS_JOINT, REL_JOINT
# relative to what?

# define different PROPRIO_OBS_TYPES
# how are orientations represented?
# what are they relative to?



class ActionSpec:
    representation = None
    reference_frame = None


class ProprioObservationSpec:
    representation = None
    reference_frame = None

