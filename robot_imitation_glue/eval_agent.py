import time

import cv2
import loguru
import numpy as np
import rerun as rr

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot_imitation_glue.base import BaseAgent, BaseDatasetRecorder
from robot_imitation_glue.utils import precise_wait

# create type for callable that takes obs and returns action


logger = loguru.logger


class State:
    rollout_active = False
    is_stopped = False
    is_paused = False


class Event:
    start_rollout = False
    stop_rollout = False
    delete_last = False
    pause = False
    resume = False
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
            if key == keyboard.Key.enter and not state.rollout_active:
                event.start_rollout = True

            elif key == keyboard.Key.enter and state.rollout_active:
                event.stop_rollout = True

            elif hasattr(key, "char") and key.char == "p" and not state.rollout_active and not state.is_paused:
                # pause the episode
                event.pause = True

            elif hasattr(key, "char") and key.char == "p" and state.is_paused:
                # resume the episode
                event.resume = True

            elif hasattr(key, "char") and key.char == "q":
                event.quit = True

            elif hasattr(key, "char") and key.char == "d" and not state.rollout_active:
                # delete the last episode
                event.delete_last = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener


def eval(  # noqa: C901
    env,
    teleop_agent: BaseAgent,
    policy_agent: BaseAgent,
    recorder: BaseDatasetRecorder,
    policy_to_pose_converter,
    teleop_to_pose_converter,
    fps=10,
    eval_dataset: LeRobotDataset = None,
    eval_dataset_image_key: str = "scene",
):
    """
    Evalulate a (policy) agent on a robot environment.

    You should also specify a teleop agent which allows to move the robot arm between policy rollouts to set the initial state.

    Rollouts are recorded using the provided dataset recorder.

    You can provide a dataset to load the initial scene image from. This is useful to evaluate the policy on a specific scene.

    Args:
        env: robot environment
        teleop_agent: teleop agent
        policy_agent: policy agent
        recorder: dataset recorder
        policy_to_pose_converter: function to convert policy action to robot pose
        teleop_to_pose_converter: function to convert teleop action to robot pose
        fps: frames per second for the dataset recorder
        eval_dataset: dataset to load initial scene image from
        eval_dataset_image_key: key in the dataset to load the image from

    """
    state = State()
    event = Event()
    listener = init_keyboard_listener(event, state)

    rr.init("robot_imitation_glue", spawn=True)

    control_period = 1 / fps
    num_rollouts = 0

    while not state.is_stopped:

        initial_scene_image = None
        instruction = None

        # load initial image from the dataset if provided. display it on top of the current scene image,
        # this allows to set the initial state of the scene.
        if eval_dataset is not None:
            n_dataset_episodes = eval_dataset.num_episodes
            if num_rollouts <= n_dataset_episodes:
                eval_dataset_episode = num_rollouts
            else:
                eval_dataset_episode = -1

            if eval_dataset_episode > -1:
                # get initial scene image
                print(eval_dataset.episode_data_index)
                step_idx = eval_dataset.episode_data_index["from"][eval_dataset_episode].item()
                initial_scene_image = eval_dataset[step_idx][eval_dataset_image_key]

                # convert to numpy array of uint8 values
                initial_scene_image = initial_scene_image.permute(1, 2, 0).numpy()
                initial_scene_image *= 255
                initial_scene_image = initial_scene_image.astype(np.uint8)

                instruction = eval_dataset[step_idx]["task"]
                logger.info(
                    f"Loading initial state of episode {eval_dataset_episode} from eval dataset with instruction: {instruction}."
                )

            if initial_scene_image is not None:
                # show initial scene image
                rr.log("initial_scene_image", rr.Image(initial_scene_image))

        target_pose = env.get_robot_pose_se3()
        target_gripper_state = env.get_gripper_opening()

        logger.info("Start teleop")
        while not state.rollout_active:
            cycle_end_time = time.time() + control_period

            observations = env.get_observations()

            vis_image = observations[eval_dataset_image_key]
            rr.log("scene", rr.Image(vis_image))
            if initial_scene_image is not None:
                # blend initial scene image with current scene image
                blended_image = cv2.addWeighted(initial_scene_image, 0.5, vis_image, 0.5, 0)
                rr.log("initial_scene_image", rr.Image(blended_image))

            action = teleop_agent.get_action(observations)
            logger.debug(f"teleop action: {action}")

            # convert teleop action to env action
            target_pose, target_gripper_state = teleop_to_pose_converter(target_pose, target_gripper_state, action)
            logger.debug(f"robot_target_pose: {target_pose}")

            env.act(
                robot_pose_se3=target_pose,
                gripper_pose=target_gripper_state,
                timestamp=time.time() + control_period,
            )

            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)

            if event.quit:
                state.is_stopped = True
                listener.stop()
                recorder.finish_recording()
                logger.info("Stop evaluation")
                return

            if event.start_rollout:
                state.rollout_active = True
            event.clear()

        logger.info("Start rollout")
        recorder.start_episode()
        while not state.is_stopped and state.rollout_active:
            cycle_end_time = time.time() + control_period

            observations = env.get_observations()

            vis_image = observations[eval_dataset_image_key].copy()
            ## print number of episodes to image
            cv2.putText(
                vis_image,
                f"Episode: {recorder.n_recorded_episodes}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            rr.log("scene", rr.Image(vis_image))

            action = policy_agent.get_action(observations)

            # convert teleop action to env action
            new_robot_target_pose, new_target_gripper_state = policy_to_pose_converter(
                target_pose, target_gripper_state, action
            )
            env.act(
                robot_pose_se3=new_robot_target_pose,
                gripper_pose=new_target_gripper_state,
                timestamp=time.time() + control_period,
            )

            recorder.record_step(observations, action)

            target_pose = new_robot_target_pose
            target_gripper_state = new_target_gripper_state

            if cycle_end_time > time.time():
                precise_wait(cycle_end_time)

            if event.stop_rollout:
                state.rollout_active = False
                num_rollouts += 1
                logger.info(f"Stop rollout {num_rollouts}")
                recorder.save_episode()
                event.clear()
                logger.info(f"Saved episode {recorder.n_recorded_episodes}")


if __name__ == "__main__":
    """example of how to use the eval function"""
    import os

    from robot_imitation_glue.dataset_recorder import LeRobotDatasetRecorder
    from robot_imitation_glue.mock import MockAgent, MockEnv, mock_agent_to_pose_converter

    env = MockEnv()
    env.reset()
    teleop_agent = MockAgent()
    policy_agent = MockAgent()

    if os.path.exists("datasets/demo"):
        dataset = LeRobotDataset(repo_id="mock", root="datasets/demo")
    else:
        dataset = None
    # create a dataset recorder

    if os.path.exists("datasets/test_dataset"):
        os.system("rm -rf datasets/test_dataset")
    dataset_recorder = LeRobotDatasetRecorder(
        example_obs_dict=env.get_observations(),
        example_action=np.zeros((7,), dtype=np.float32),
        root_dataset_dir="datasets/test_dataset",
        dataset_name="test_dataset",
        fps=10,
        use_videos=True,
    )

    eval(
        env,
        teleop_agent,
        policy_agent,
        dataset_recorder,
        mock_agent_to_pose_converter,
        mock_agent_to_pose_converter,
        fps=2,
        eval_dataset=dataset,
    )
