import abc


class ActionSpec:
    representation = None
    reference_frame = None


class ProprioObservationSpec:
    representation = None
    reference_frame = None


class BaseEnv(abc.ABC):
    ACTION_SPEC = None
    PROPRIO_OBS_SPEC = None

    @abc.abstractmethod
    def get_observations(self):
        """ observations as a dict of (str, np.ndarray)"""
        pass

    @abc.abstractmethod
    def act(self, robot_pose_se3, gripper_pose, timestamp):
        raise NotImplementedError

    def reset(self):
        pass

    @abc.abstractmethod
    def get_joint_configuration(self):
        """ joint configuration as a (n,) numpy array"""
        pass

    @abc.abstractmethod
    def get_robot_pose_se3(self):
        """ robot pose in base frame as a 4x4 numpy array"""
        pass

    @abc.abstractmethod
    def get_gripper_opening(self):
        """ absolute gripper opening in meters as a (1,) numpy array"""
        pass

    @property
    def action_spec(self):
        return self.ACTION_SPEC

    @property
    def proprio_obs_spec(self):
        return self.PROPRIO_OBS_SPEC


class BaseAgent:
    ACTION_SPEC = None

    def get_action(self, observation):
        pass

    @property
    def action_spec(self):
        return self.ACTION_SPEC


# define different action types
# ABS_EEF, REL_EEF, ABS_JOINT, REL_JOINT
# relative to what?

# define different PROPRIO_OBS_TYPES
# how are orientations represented?
# what are they relative to?


class BaseDatasetRecorder:
    def __init__(self):
        pass

    def start_episode(self):
        raise NotImplementedError

    def record_step(self, obs, action):
        raise NotImplementedError

    def save_episode(self):
        raise NotImplementedError

    def finish_recording(self):
        pass

    @property
    def n_recorded_episodes(self):
        pass
