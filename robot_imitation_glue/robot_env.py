"""
Environement for Robot Station with 2 realsense cameras and a UR3e robot with a Robotiq Gripper

Actions as absolute target pose (rotation vector) in robot base frame and absolute gripper width
Proprioception as robot pose (euler angles) in robot base frame and gripper width
"""

import time

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_robots.grippers.hardware.robotiq_2f85_urcap import Robotiq2F85
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_spatial_algebra.se3 import SE3Container, normalize_so3_matrix

from robot_imitation_glue.base import BaseEnv
from robot_imitation_glue.ipc_camera import RGBCameraPublisher, RGBCameraSubscriber, initialize_ipc

# env consists of 2 realsense cameras and UR3e robot

WRIST_REALSENSE_SERIAL = "925322060348"
WRIST_CAM_RGB_TOPIC = "wrist_rgb"
WRIST_CAM_RESOLUTION_TOPIC = "wrist_resolution"

SCENE_REALSENSE_SERIAL = "231122072220"
SCENE_CAM_RGB_TOPIC = "scene_rgb"
SCENE_CAM_RESOLUTION_TOPIC = "scene_resolution"

ROBOT_IP = "10.42.0.162"

logger = loguru.logger


class CameraFactory:
    def create_wrist_camera():
        return Realsense(resolution=Realsense.RESOLUTION_480, fps=30, serial_number=WRIST_REALSENSE_SERIAL)

    def create_scene_camera():
        return Realsense(resolution=Realsense.RESOLUTION_480, fps=30, serial_number=SCENE_REALSENSE_SERIAL)


class UR3eStation(BaseEnv):
    ACTION_SPEC = None
    PROPRIO_OBS_SPEC = None

    def __init__(self):
        # set up cameras
        initialize_ipc()

        logger.info("Creating wrist camera publisher.")
        self._wrist_camera_publisher = RGBCameraPublisher(
            CameraFactory.create_wrist_camera,
            WRIST_CAM_RGB_TOPIC,
            WRIST_CAM_RESOLUTION_TOPIC,
            100,
        )
        self._wrist_camera_publisher.start()

        logger.info("Creating wrist camera subscriber.")
        self._wrist_camera_subscriber = RGBCameraSubscriber(
            WRIST_CAM_RESOLUTION_TOPIC,
            WRIST_CAM_RGB_TOPIC,
        )

        logger.info("Creating scene camera publisher.")
        self._scene_camera_publisher = RGBCameraPublisher(
            CameraFactory.create_scene_camera,
            SCENE_CAM_RGB_TOPIC,
            SCENE_CAM_RESOLUTION_TOPIC,
            100,
        )
        self._scene_camera_publisher.start()

        logger.info("Creating scene camera subscriber.")
        self._scene_camera_subscriber = RGBCameraSubscriber(
            SCENE_CAM_RESOLUTION_TOPIC,
            SCENE_CAM_RGB_TOPIC,
        )

        # self._wrist_camera_subscriber = self._scene_camera_subscriber
        # wait for first images
        time.sleep(2)

        # set up robot and gripper
        logger.info("connecting to gripper.")
        self.gripper = Robotiq2F85(ROBOT_IP)

        logger.info("connecting to robot.")
        self.robot = URrtde(ROBOT_IP, URrtde.UR3E_CONFIG, gripper=self.gripper)

        # set up additional sensors if needed.

        # rr.init("ur3e-station",spawn=True)

    def get_joint_configuration(self):
        return self.robot.get_joint_configuration()

    def get_robot_pose_euler(self):
        """
        pose as [x,y,z,rx,ry,rz] in robot base frame using Euler angles
        """
        hom_pose = self.robot.get_tcp_pose()
        rotation_vector = SE3Container.from_homogeneous_matrix(hom_pose).orientation_as_euler_angles
        position = hom_pose[:3, 3]
        return np.concatenate((position, rotation_vector), axis=0)

    def get_robot_pose_se3(self):
        return self.robot.get_tcp_pose()

    def get_gripper_opening(self):
        return np.array([self.gripper.get_current_width()])

    def _set_robot_target_pose(self, target_pose):
        # target_pose is a 4x4 homogeneous transformation matrix
        self.robot.servo_to_tcp_pose()

    def get_observations(self):

        wrist_image = self._wrist_camera_subscriber.get_rgb_image_as_int()
        scene_image = self._scene_camera_subscriber.get_rgb_image_as_int()
        robot_state = self.get_robot_pose_euler().astype(np.float32)
        gripper_state = self.get_gripper_opening().astype(np.float32)
        joints = self.robot.get_joint_configuration().astype(np.float32)

        state = np.concatenate((robot_state, gripper_state), axis=0)
        # TODO: resize images (but still include the original?)

        # resize wrist images to 224x224
        wrist_image_resized = cv2.resize(wrist_image, (224, 224))
        # resize scene img, cut first 200 x pixels
        scene_image_resized = scene_image.copy()
        scene_image_resized = scene_image_resized[:, 200:]
        # resize scene image to 224x224
        scene_image_resized = cv2.resize(scene_image_resized, (224, 224))

        obs_dict = {
            "wrist_image_original": wrist_image,
            "scene_image_original": scene_image,
            "wrist_image": wrist_image_resized,
            "scene_image": scene_image_resized,
            "state": state,
            "robot_pose": robot_state,
            "gripper_state": gripper_state,
            "joints": joints,
        }

        # add to rerun
        # rr.log("wrist",rr.Image(wrist_image))
        # rr.log("scene",rr.Image(scene_image))

        return obs_dict

    def act(self, robot_pose_se3, gripper_pose, timestamp):

        if isinstance(gripper_pose, np.ndarray):
            gripper_pose = gripper_pose[0].item()

        # move robot to target pose
        current_time = time.time()
        duration = timestamp - current_time
        if duration < 0:
            logger.warning("Action duration is negative, setting it to 0")
            duration = 0
        logger.debug(f"Moving robot to pose {robot_pose_se3} with duration {duration}")

        robot_pose_se3[:3, :3] = normalize_so3_matrix(robot_pose_se3[:3, :3])
        self.robot.servo_to_tcp_pose(robot_pose_se3, duration)

        # move gripper to target width
        gripper_width = np.clip(
            gripper_pose, self.gripper.gripper_specs.min_width, self.gripper.gripper_specs.max_width
        )
        self.gripper._set_target_width(gripper_width)

        # do not wait, handling timings is the responsibility of the caller
        return

    def close(self):
        self._wrist_camera_publisher.stop()
        self._scene_camera_publisher.stop()


if __name__ == "__main__":
    # set cli logging level to debug

    def convert_relative_to_absolute_action(current_robot_pose, current_gripper_width, action: np.ndarray):
        """
        Convert a relative action to an absolute action
        Args:
            action: [x,y,z, rx,ry,rz,gripper]. relative target pose in robot base frame (euler angles) and absolute gripper width
            current_robot_pose: [x,y,z, rx,ry,rz]. current robot pose in robot base frame using Euler angles
            current_gripper_width: current gripper width

        Returns:
            [x,y,z, rx,ry,rz,gripper]. absolute target pose in robot base frame (rotation vector) and absolute gripper width
        """
        current_robot_pose_se3 = SE3Container.from_euler_angles_and_translation(
            current_robot_pose[3:6], current_robot_pose[0:3]
        ).homogeneous_matrix

        relative_robot_pose = SE3Container.from_euler_angles_and_translation(
            action[3:6], action[0:3]
        ).homogeneous_matrix
        target_pose_se3 = current_robot_pose_se3 @ relative_robot_pose

        target_gripper_width = action[6] + current_gripper_width

        return target_pose_se3, target_gripper_width

    env = UR3eStation()
    cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
    cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wrist", 640, 480)
    cv2.resizeWindow("scene", 640, 480)

    direction = 1
    z_actions = []
    z_positions = []
    while True:
        obs = env.get_observations()
        print(time.time())
        print(obs["wrist_image"].shape)
        cv2.imshow("wrist", obs["wrist_image"])
        cv2.imshow("scene", obs["scene_image"])

        current_pose = obs["robot_pose"]
        current_z = current_pose[2]
        print(current_z)
        if current_z > 0.2:
            direction = 1
        elif current_z < 0.1:
            direction = -1
        z_action = 0.02 * direction
        gripper_action = 0.005 if direction == 1 else -0.005
        action = np.array([0, 0, z_action, 0, 0, z_action, 0])
        logger.debug(f"Taking action {action}")
        logger.debug(f"Current pose {current_pose}")

        robot_se3, gripper = convert_relative_to_absolute_action(current_pose, obs["gripper_state"], action)
        env.act(robot_pose_se3=robot_se3, gripper_pose=gripper, timestamp=time.time() + 0.1)
        key = cv2.waitKey(100)
