from robot_imitation_glue.base import BaseAgent, BaseDatasetRecorder, BaseEnv
from robot_imitation_glue.utils import precise_wait

env = None
agent = None

import time

import cv2
import loguru
import numpy as np

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

            elif hasattr(key, "char") and key.char == "q":
                event.quit = True

            elif hasattr(key, "char") and key.char == "d" and not state.is_recording:
                # delete the last episode
                event.delete_last = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener


# create type for callable that takes obs and returns action
from typing import Callable

converter_callable = Callable[dict[str, np.ndarray], np.ndarray]


def collect_data(  # noqa: C901
    env: BaseEnv,
    teleop_agent: BaseAgent,
    dataset_recorder: BaseDatasetRecorder,
    frequency=10,
    teleop_to_pose_converter: converter_callable = None,
    abs_pose_to_policy_action: converter_callable = None,
):
    assert env.ACTION_SPEC == teleop_agent.ACTION_SPEC

    # create cv2 window as GUI.
    cv2.namedWindow("robot_imitation_glue", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("robot_imitation_glue", 640, 480)

    # TODO: based on provided path, create new dataset or load existing dataset

    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    control_period = 1 / frequency

    target_pose = env.get_robot_pose_se3()
    target_gripper_state = env.get_gripper_opening()

    while not state.is_stopped:
        cycle_end_time = time.time() + control_period

        before_observation_time = time.time()
        observation = env.get_observations()
        after_observation_time = time.time()
        after_observation_time - before_observation_time
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
            # TODO: allow for textual description of the episode?

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
        vis_img = observation["scene_image"].copy()
        print(vis_img)
        cv2.imshow("robot_imitation_glue", vis_img)
        # visualize state is_recording, is_paused
        cv2.putText(
            vis_img, f"recording = {state.is_recording}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(vis_img, f"Paused: {state.is_paused}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(
            vis_img,
            f"num episodes collected: {dataset_recorder.n_recorded_episodes}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # if paused, do not collect teleop or execute action
        if state.is_paused:
            time.sleep(0.1)
            continue

        action = teleop_agent.get_action(observation)
        logger.info(f"Action: {action}")

        new_robot_target_se3_pose, new_gripper_target_width = teleop_to_pose_converter(
            target_pose, target_gripper_state, action
        )

        policy_formatted_action = abs_pose_to_policy_action(
            target_pose, target_gripper_state, new_robot_target_se3_pose, new_gripper_target_width
        )

        env.act(
            robot_pose_se3=new_robot_target_se3_pose,
            gripper_pose=new_gripper_target_width,
            timestamp=time.time() + control_period,
        )

        if state.is_recording:
            dataset_recorder.record_step(observation, policy_formatted_action)

        # wait for end of the control period
        if cycle_end_time > time.time():
            precise_wait(cycle_end_time)
        else:
            print("cycle time exceeded control period")

        # update the target pose and target gripper state for the next iteration
        target_pose = new_robot_target_se3_pose
        target_gripper_state = new_gripper_target_width

        # TODO: we now use 'integration' to get the next target pose instead of using the current pose.
        # this is to avoid 'shaking' of the robot, as is done in diffusion policy teleop for example.
        # but need to verify that this does not cause mismatch between teleop and policy.
        # and should also check if the distance between the target and the actual robot does not diverge too much.


if __name__ == "__main__":
    # create dummy env, agent and recorder to test flow.
    from scipy.spatial.transform import Rotation as R

    from robot_imitation_glue.dataset_recorder import DummyDatasetRecorder
    from robot_imitation_glue.robot_env import UR3eStation
    from robot_imitation_glue.spacemouse_agent import SpaceMouseAgent

    env = UR3eStation()

    def delta_action_to_abs_se3_converter(robot_pose_se3, gripper_state, action):
        # convert spacemouse action to ur3e action
        # we take the action to consist of a delta position, delta rotation and delta gripper width.
        # the delta rotation is interpreted as expressed in a frame with the same orientation as the base frame but with the origin at the EEF.
        # in this way, when rotating the spacemouse, the robot eef will not move around, while at the same time the axes of orientation
        # do not depend on the current orientation of the EEF.

        # the delta position is intepreted in the world frame and also applied on the EEF frame.

        delta_pos = action[:3]
        delta_rot = action[3:6]
        gripper_action = action[6]

        robot_trans = robot_pose_se3[:3, 3]
        robot_SO3 = robot_pose_se3[:3, :3]

        new_robot_trans = robot_trans + delta_pos
        # rotation is now interpreted as euler and not as rotvec
        # similar to Diffusion Policy.
        # however, rotvec seems more principled (related to twist)
        new_robot_SO3 = R.from_euler("xyz", delta_rot).as_matrix() @ robot_SO3

        new_robot_SE3 = np.eye(4)
        new_robot_SE3[:3, :3] = new_robot_SO3
        new_robot_SE3[:3, 3] = new_robot_trans

        new_gripper_state = gripper_state + gripper_action
        new_gripper_state = np.clip(new_gripper_state, 0, 0.085)

        return new_robot_SE3, new_gripper_state

    def abs_se3_to_relative_policy_action_converter(robot_pose, gripper_pose, abs_se3_action, gripper_action):
        relative_se3 = np.linalg.inv(robot_pose) @ abs_se3_action

        relative_pos = relative_se3[:3, 3]
        relative_euler = R.from_matrix(relative_se3[:3, :3]).as_euler("xyz")
        relative_gripper = gripper_action - gripper_pose

        return np.concatenate((relative_pos, relative_euler, relative_gripper), axis=0)

    agent = SpaceMouseAgent()
    dataset_recorder = DummyDatasetRecorder()

    collect_data(
        env,
        agent,
        dataset_recorder,
        10,
        delta_action_to_abs_se3_converter,
        abs_se3_to_relative_policy_action_converter,
    )
