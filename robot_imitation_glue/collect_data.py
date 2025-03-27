from robot_imitation_glue.utils import precise_wait
from robot_imitation_glue.base import BaseEnv, BaseDatasetRecorder, BaseAgent
env = None 
agent = None 

import cv2
import numpy as np
import time 
import loguru

logger = loguru.logger


class State:
    is_recording = False
    is_stopped = False
    is_paused = False

class Event:
    start_recording = False
    stop_recording = False
    delete_last = False
    pause = False
    resume = False
    delete_last = False
    quit = False

    def clear(self):
        for attr in self.__dict__:
            setattr(self, attr, False)

    
def init_keyboard_listener(event: Event, state: State):
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            # "space bar"
            if key == keyboard.Key.space and not state.is_recording:
                event.start_recording = True

            elif key == keyboard.Key.space and state.is_recording:
                event.stop_recording = True

            elif key == keyboard.Key.enter and not state.is_recording and not state.is_paused:
                # pause the episode
                event.pause = True
            
            elif key == keyboard.Key.enter and state.is_paused:
                # resume the episode
                event.resume = True
            

            elif hasattr(key,'char') and key.char == 'q':
                event.quit = True

            elif hasattr(key,'char') and key.char == 'd' and not state.is_recording:
                # delete the last episode
                event.delete_last = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener


# create type for callable that takes obs and returns action
from typing import Callable

converter_callable = Callable[dict[str,np.ndarray], np.ndarray]


def collect_data(env: BaseEnv, teleop_agent: BaseAgent, dataset_recorder: BaseDatasetRecorder, frequency=10, teleop_to_policy_converter: converter_callable = None, teleop_to_env_converter: converter_callable = None):
    assert env.ACTION_SPEC == teleop_agent.ACTION_SPEC

    # create cv2 window as GUI.
    cv2.namedWindow("robot_imitation_glue", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("robot_imitation_glue", 640, 480)


    # 

    
    # based on provided path, create new dataset or load existing dataset

    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)


    control_period = 1 / frequency
    while not state.is_stopped:
        cycle_end_time = time.time() +  control_period

        before_observation_time = time.time()
        observation = env.get_observations()
        after_observation_time = time.time()
        observation_time = after_observation_time - before_observation_time
        # print("observation time: ", observation_time)


 
        # update & handle state machine events
        if not state.is_recording and event.start_recording:
            state.is_recording = True
            print("start recording")
            dataset_recorder.start_episode()

        elif state.is_recording and event.stop_recording:
            state.is_recording = False
            print("stop recording")
            # save episode
            dataset_recorder.save_episode()
            #TODO: allow for textual description of the episode?

        elif event.delete_last and not state.is_recording:
            print("delete last episode")
            # delete last episode
        
        elif event.pause and not state.is_recording:
            state.is_paused = True
            print("pause teleop")
        
        elif event.resume and state.is_paused:
            state.is_paused = False
            print("resume teleop")

        elif event.quit:
            print("quit")
            state.is_stopped = True
            listener.stop()
            dataset_recorder.finish_recording()
            break

        # clear all events
        event.clear()
        

        # update GUI.
        #TODO: cv2 window.


        # if paused, do not collect teleop or execute action
        if state.is_paused:
            time.sleep(0.1)
            continue

        action = teleop_agent.get_action(observation)
        logger.info(f"Action: {action}")

        if teleop_to_policy_converter:
            policy_formatted_action = teleop_to_policy_converter(observation, action)
        else:
            policy_formatted_action = action
        
        if teleop_to_env_converter:
            env_formatted_action = teleop_to_env_converter(observation, action)
        else:
            env_formatted_action = action

        env.act(env_formatted_action,time.time() + control_period)
        
        if state.is_recording:
            dataset_recorder.record_step(observation, policy_formatted_action)

        # wait for end of the control period
        if cycle_end_time > time.time():
            precise_wait(cycle_end_time)
        else:
            print("cycle time exceeded control period")

    




if __name__ =="__main__":
    # create dummy env, agent and recorder to test flow.
    from robot_imitation_glue.robot_env import UR3eStation
    from robot_imitation_glue.spacemouse_agent import SpaceMouseAgent
    from robot_imitation_glue.dataset_recorder import DummyDatasetRecorder
    from robot_imitation_glue.robot_env import convert_absolute_to_relative_action
    from scipy.spatial.transform import Rotation as R

    env = UR3eStation()

    def spacemouse_to_ur3e_converter(observation, action):
        # convert spacemouse action to ur3e action
        robot_pose = observation["robot_pose"]
        gripper_state = observation["gripper_state"]
        
        twist_trans = action[:3]
        twist_rot = action[3:6]
        gripper_action = action[6]

        robot_trans = robot_pose[:3]
        robot_euler = robot_pose[3:6]

        new_robot_trans = robot_trans + twist_trans 
        new_robot_rotvec = (R.from_euler('xyz',twist_rot) * R.from_euler('xyz', robot_euler)).as_rotvec() 


        new_gripper_state = gripper_state + gripper_action
        return np.concatenate([new_robot_trans, new_robot_rotvec, new_gripper_state])


    def spacemouse_to_relative_policy_converter(observation, action):
        abs = spacemouse_to_ur3e_converter(observation, action)
        current_robot_pose = observation["robot_pose"]
        current_gripper_state = observation["gripper_state"]
        return convert_absolute_to_relative_action(current_robot_pose, current_gripper_state, abs)
    
    agent = SpaceMouseAgent()
    dataset_recorder = DummyDatasetRecorder()

    collect_data(env, agent, dataset_recorder,10, spacemouse_to_relative_policy_converter, spacemouse_to_ur3e_converter)
