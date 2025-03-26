"""This file implements a multiprocess pub-sub for an airo-mono RGB camera.

This requires you to install the airo-camera-toolkit, which you can do by following the instructions here:
https://github.com/airo-ugent/airo-mono
"""
import time
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
from airo_camera_toolkit.cameras.opencv_videocapture.opencv_videocapture import OpenCVVideoCapture
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import CameraIntrinsicsMatrixType, NumpyIntImageType, NumpyFloatImageType, CameraResolutionType
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from loguru import logger

from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIDL
from airo_ipc.cyclone_shm.patterns.ddsreader import DDSReader
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader
from airo_ipc.framework.framework import initialize_ipc, IpcKind
from airo_ipc.framework.node import Node




@dataclass
class ResolutionIdl(IdlStruct):
    """We will send the resolution of the webcam over DDS: we need to define an IDL struct for this."""
    width: int
    height: int


@dataclass
class RGBFrame(BaseIDL):
    """We will send the RGB frames over shared memory: we need to derive from BaseIDL."""
    timestamp: np.ndarray
    rgb: np.ndarray
    intrinsics: np.ndarray


 
    @staticmethod
    def with_resolution(width: int, height: int):
        """We may not know the resolution of the webcam when we create the frame, so we need a factory method."""
        return RGBFrame(rgb=np.zeros((height, width, 3), dtype=np.uint8), intrinsics=np.zeros((3, 3)),timestamp=np.zeros((1,)))


class RGBCameraPublisher(Node):
    """The publisher will open the webcam and publish the resolution and frame in a loop."""
    def __init__(self, camera_creation_fn, rgb_topic_name, resolution_topic_name, update_frequency, verbose = False):
        self._camera_creation_fn = camera_creation_fn
        self._camera: Optional[RGBCamera] = None
        self._rgb_topic_name = rgb_topic_name
        self._resolution_topic_name = resolution_topic_name


        super().__init__(update_frequency, verbose)
    def _setup(self):
        logger.info("Opening camera.")
        self._camera = self._camera_creation_fn()
        assert isinstance(self._camera, RGBCamera), "Camera creation function must return an instance of RGBCamera"

        logger.info("Getting resolution.")
        width, height = self._camera.resolution

        logger.info("Registering publishers.")
        self._register_publisher(self._resolution_topic_name, ResolutionIdl, IpcKind.DDS)
        self._register_publisher(self._rgb_topic_name, RGBFrame.with_resolution(width, height), IpcKind.SHARED_MEMORY)

    def _step(self):
        """The _step method is called in a loop by the Node superclass."""
        rgb = self._camera.get_rgb_image_as_int()

        self._publish(self._resolution_topic_name,
                      ResolutionIdl(width=self._camera.resolution[0], height=self._camera.resolution[1]))
        self._publish(self._rgb_topic_name, RGBFrame(rgb=rgb, intrinsics=self._camera.intrinsics_matrix(),timestamp=np.array([time.time()])))

    def _teardown(self):
            pass

class RGBCameraSubscriber(RGBCamera):

    def __init__(self, resolution_topic: str, rgb_topic: str):
        super().__init__()

        self._cyclone_dp = DomainParticipant()
        self._reader_resolution = DDSReader(self._cyclone_dp, resolution_topic, ResolutionIdl)
        # Wait for the first resolution message.
        resolution = None
        while resolution is None:
            resolution = self._reader_resolution()
            logger.info("Did not yet receive resolution message. Sleeping for 100 milliseconds...")
            time.sleep(0.1)
        self._reader_rgb = SMReader(self._cyclone_dp, rgb_topic,
                                    RGBFrame.with_resolution(resolution.width, resolution.height))

    @property
    def resolution(self) -> CameraResolutionType:
        return self._resolution

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._rgb).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._rgb

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_matrix
    
    def get_timestamp(self) -> float:
        """Get the timestamp of the current image."""
        return self._timestamp

    def _grab_images(self) -> None:
        frame = self._reader_rgb()
        if frame is not None:
            self._rgb = frame.rgb  # Already copied in the SMReader.
            self._intrinsics_matrix = frame.intrinsics
        
            self._timestamp = frame.timestamp[0].item()
    

class CameraFactory:
    def create_camera():
        #return OpenCVVideoCapture(resolution=(1920, 1080),  fps=30,intrinsics_matrix=np.eye(3))
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense
        return Realsense(resolution=Realsense.RESOLUTION_1080,fps=30)



if __name__ == '__main__':
    import cv2
    initialize_ipc()

    TOPIC_RGB = "webcam_rgb"
    TOPIC_RESOLUTION = "webcam_resolution"
    logger.info("Creating publisher.")

    publisher = RGBCameraPublisher(CameraFactory.create_camera,TOPIC_RGB, TOPIC_RESOLUTION,100, True)
    logger.info("Starting publisher.")
    publisher.start()

    logger.info("Creating subscriber.")
    subscriber = RGBCameraSubscriber(TOPIC_RESOLUTION, TOPIC_RGB)

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)

    while True:
        rgb = subscriber.get_rgb_image_as_int()
        image_timestamp = subscriber.get_timestamp()
        current_time = time.time()
        logger.debug(f"Timestamp: {image_timestamp}, Current time: {current_time}, Diff: {current_time - image_timestamp}")
        rgb_cv = ImageConverter.from_numpy_int_format(rgb).image_in_opencv_format

        cv2.imshow("Webcam", rgb_cv)
        key = cv2.waitKey(1)
        if key == ord("q"):
            logger.info("Stopping...")
            break

    publisher.stop()
    cv2.destroyAllWindows()