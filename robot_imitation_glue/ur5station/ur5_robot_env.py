"""
Environement for Robot Station with 2 realsense cameras and a UR3e robot with a Robotiq Gripper

Actions as absolute target pose (rotation vector) in robot base frame and absolute gripper width
Proprioception as robot pose (euler angles) in robot base frame and gripper width
"""

import os
import time

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.realsense.realsense import Realsense
from airo_camera_toolkit.cameras.zed.zed import Zed
from airo_robots.manipulators.hardware.ur_rtde import URrtde
from airo_spatial_algebra.se3 import SE3Container, normalize_so3_matrix

from robot_imitation_glue.base import BaseEnv
from robot_imitation_glue.grippers.schunk_process import SchunkGripperProcess
from robot_imitation_glue.ipc_camera import RGBCameraPublisher, RGBCameraSubscriber, initialize_ipc

# env consists of 1 zed scene camera, 1 wrist  realsense cameras and a UR5e robot + Schunk gripper

WRIST_REALSENSE_SERIAL = "817612070315"
WRIST_CAM_RGB_TOPIC = "wrist_rgb"
WRIST_CAM_RESOLUTION_TOPIC = "wrist_resolution"

SCENE_ZED_SERIAL = "31733653"
SCENE_CAM_RGB_TOPIC = "scene_rgb"
SCENE_CAM_RESOLUTION_TOPIC = "scene_resolution"

ROBOT_IP = "10.42.0.163"

SCHUNK_GRIPPER_HOST = "/dev/ttyUSB0,11,115200,8E1"  # run bks_scan -H <usb> to find the slaveID, run dmesg | grep tty to find the usb port

HOME_JOINTS = np.array([-180, -90, 90, -90, -90, -90]) * np.pi / 180  # for left UR5e on dual arm setup in mano lab.

logger = loguru.logger


class CameraFactory:
    def create_wrist_camera():
        return Realsense(resolution=Realsense.RESOLUTION_480, fps=30, serial_number=WRIST_REALSENSE_SERIAL)

    def create_scene_camera():
        return Zed(
            resolution=Zed.RESOLUTION_VGA, fps=30, depth_mode=Zed.NONE_DEPTH_MODE, serial_number=SCENE_ZED_SERIAL
        )


class UR5eStation(BaseEnv):
    ACTION_SPEC = None
    PROPRIO_OBS_SPEC = None

    def __init__(self):

        # set up robot and gripper
        # logger.info("connecting to gripper.")

        # set environment variable for bks gripper comm
        os.environ["BKS_HOST"] = SCHUNK_GRIPPER_HOST
        self.gripper = SchunkGripperProcess(SCHUNK_GRIPPER_HOST)
        time.sleep(2)
        self.gripper.max_grasp_force = self.gripper.gripper_specs.min_force  # minimal force for EGK40 is 55N
        self.gripper.speed = self.gripper.gripper_specs.max_speed
        logger.info("connecting to robot.")
        self.robot = URrtde(ROBOT_IP, URrtde.UR3E_CONFIG, gripper=self.gripper)

        robot_awaitable = self.robot.move_to_joint_configuration(
            HOME_JOINTS
        )  # do not wait, let cameras initialize first

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

        self._wrist_camera_subscriber = RGBCameraSubscriber(
            WRIST_CAM_RESOLUTION_TOPIC,
            WRIST_CAM_RGB_TOPIC,
        )
        # wait for first images
        time.sleep(2)

        robot_awaitable.wait()

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

        start_time = time.time()
        wrist_image = self._wrist_camera_subscriber.get_rgb_image_as_int()
        scene_image = self._scene_camera_subscriber.get_rgb_image_as_int()
        robot_state = self.get_robot_pose_euler().astype(np.float32)
        gripper_state = self.get_gripper_opening().astype(np.float32)
        joints = self.robot.get_joint_configuration().astype(np.float32)

        state = np.concatenate((robot_state, gripper_state), axis=0)

        # resize wrist images to 224x224
        wrist_image_resized = cv2.resize(wrist_image, (224, 224))
        scene_image_resized = scene_image.copy()
        # resize scene image to 224x224
        scene_image_resized = cv2.resize(scene_image_resized, (224, 224), interpolation=cv2.INTER_CUBIC)

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
        logger.info(f"get_observations time: {time.time() - start_time}")

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

        z_coord = robot_pose_se3[2, 3]

        if z_coord < 0.0:
            # too far
            logger.warning("Z coordinate is below zero . not executing action")
            return

        self.robot.servo_to_tcp_pose(robot_pose_se3, duration)

        # move gripper to target width
        gripper_width = np.clip(
            gripper_pose, self.gripper.gripper_specs.min_width, self.gripper.gripper_specs.max_width
        )

        logger.debug(f"Setting gripper width to {gripper_width}")
        time_before_gripper = time.time()
        self.gripper.servo(gripper_width)
        time_after_gripper = time.time()
        logger.debug(f"Gripper servo time: {time_after_gripper - time_before_gripper}")

        # do not wait, handling timings is the responsibility of the caller
        return

    def close(self):
        self._wrist_camera_publisher.stop()
        self._scene_camera_publisher.stop()
        self.gripper.shutdown()


def convert_abs_gello_actions_to_se3(action: np.ndarray):
    tcp_pose = np.eye(4)
    tcp_pose[2, 3] = 0.176
    joints = action[:6]
    gripper = action[6]
    gripper = (1 - gripper) * 0.08  # convert to stroke width
    pose = ur5e.forward_kinematics_with_tcp(*joints, tcp_pose)
    return pose, gripper


if __name__ == "__main__":
    # set cli logging level to debug
    from ur_analytic_ik import ur5e

    from robot_imitation_glue.agents.gello import DynamixelConfig, GelloAgent

    env = UR5eStation()

    config = DynamixelConfig(
        joint_ids=[1, 2, 3, 4, 5, 6],
        joint_offsets=(np.array([40, 16, 25, 40, 15, 7]) * np.pi / 16).tolist(),
        joint_signs=[1, 1, -1, 1, 1, 1],
        gripper_config=(7, 194, 152),
    )
    agent = GelloAgent(config, "/dev/ttyUSB1")

    cv2.namedWindow("wrist", cv2.WINDOW_NORMAL)
    cv2.namedWindow("scene", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("wrist", 640, 480)
    cv2.resizeWindow("scene", 640, 480)

    input("Press Enter to start teleoperation")
    action = agent.get_action(env.get_observations())
    robot_se3, gripper = convert_abs_gello_actions_to_se3(action)
    env.robot.servo_to_tcp_pose(robot_se3, 1.0)

    while True:
        loop_time = time.time()
        obs = env.get_observations()
        print(time.time())
        print(obs["wrist_image"].shape)
        cv2.imshow("wrist", obs["wrist_image"])
        cv2.imshow("scene", obs["scene_image"])

        action = agent.get_action(obs)
        robot_se3, gripper = convert_abs_gello_actions_to_se3(action)
        env.act(robot_pose_se3=robot_se3, gripper_pose=gripper, timestamp=time.time() + 0.1)
        print(obs["state"])

        loop_duration = time.time() - loop_time
        # wait for 100ms - loop time
        key = cv2.waitKey(max(1, int(100 - loop_duration * 1000)))

    env.robot.gripper.move(0.04).wait()
    time.sleep(5)
    env.robot.gripper.move(0.0).wait()
