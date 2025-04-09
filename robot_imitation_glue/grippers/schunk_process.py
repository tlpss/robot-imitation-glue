import logging
import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs
from bkstools.bks_lib.bks_module import BKSModule
from pyschunk.generated.generated_enums import eCmdCode

logger = logging.getLogger(__name__)


def rescale_range(x: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
    return to_min + (x - from_min) / (from_max - from_min) * (to_max - to_min)


@dataclass
class GripperCommand:
    """Command sent to the gripper process"""

    cmd_type: str
    params: Dict[str, Any] = None


# values obtained from https://schunk.com/be/nl/grijpsystemen/parallelgrijper/egk/egk-40-mb-m-b/p/000000000001491762
SCHUNK_DEFAULT_SPECS = ParallelPositionGripperSpecs(0.0829, 0.0, 150, 55, 0.0575, 0.0055)


class SchunkGripperProcess(ParallelPositionGripper):
    """
    A process-based implementation of the Schunk gripper that allows continuous position polling
    while providing command execution capabilities to the parent process.
    """

    def __init__(
        self, usb_interface: str = "/dev/ttyUSB0,11", gripper_specs: Optional[ParallelPositionGripperSpecs] = None
    ):
        # Default specs for Schunk gripper if none provided
        if gripper_specs is None:
            gripper_specs = SCHUNK_DEFAULT_SPECS

        super().__init__(gripper_specs)

        # Create shared memory for gripper state
        self._position = mp.Value("f", 0.0)

        # Command queue for parent->child communication
        self._cmd_queue = mp.Queue()

        self._result_queue = mp.Queue(maxsize=1)

        # Flag to signal process termination
        self._terminate = mp.Event()

        # Create and start process
        self._process = mp.Process(
            target=self._gripper_process_main,
            args=(usb_interface, self._position, self._cmd_queue, self._result_queue, self._terminate),
        )
        self._process.daemon = True
        self._process.start()

        # Wait for gripper to initialize
        time.sleep(1.0)

    def _gripper_process_main(
        self,
        usb_interface: str,
        position,
        cmd_queue: mp.Queue,
        result_queue: mp.Queue,
        terminate,
    ):
        """Main function running in the child process"""
        # Initialize gripper
        gripper = BKSModule(usb_interface)

        # Prepare gripper: Acknowledge any pending errors
        gripper.command_code = eCmdCode.CMD_ACK
        gripper.MakeReady()
        time.sleep(0.1)

        # Main loop
        last_poll_time = time.time()
        cmd_executing = False
        while not terminate.is_set():
            # Check if there are any commands to execute
            if not cmd_queue.empty() and not cmd_executing:
                cmd_executing = True
                assert result_queue.empty()
                # get latest command from queue
                cmd = cmd_queue.get()
                try:
                    result = self._execute_command(gripper, cmd)
                    self._result_queue.put({"success": True, "result": result})
                except Exception as e:
                    result_queue.put({"success": False, "error": str(e)})

                cmd_executing = False

            # Poll gripper position (if we're not executing a command)
            current_time = time.time()
            if not cmd_executing and (current_time - last_poll_time) > 0.05:  # 20Hz polling
                with position.get_lock():
                    position.value = gripper.actual_pos
                last_poll_time = current_time

            # Small sleep to prevent CPU hogging
            time.sleep(0.001)

    def _execute_command(self, gripper: BKSModule, command: GripperCommand) -> Any:  # noqa: C901
        """Execute a command on the gripper"""
        logger.info(f"Executing command: {command.cmd_type}")
        if command.cmd_type == "move_pos":

            pos = command.params.get("position")

            gripper.MakeReady()

            gripper.set_pos = pos
            gripper.command_code = eCmdCode.MOVE_POS

            print(pos)
            print(f"Executed command: {command.cmd_type}")
            return True

        elif command.cmd_type == "grip":
            force = command.params.get("force", 50)
            gripper.set_force = force
            gripper.set_vel = 0.0
            gripper.grp_dir = True
            gripper.command_code = eCmdCode.MOVE_FORCE
            return True

        elif command.cmd_type == "stop":
            gripper.command_code = eCmdCode.CMD_STOP
            return True

        elif command.cmd_type == "fast_stop":
            gripper.command_code = eCmdCode.CMD_FAST_STOP
            return True

        elif command.cmd_type == "make_ready":
            gripper.command_code = eCmdCode.CMD_ACK
            gripper.MakeReady()
            return True

        elif command.cmd_type == "set_vel":
            vel = command.params.get("velocity")
            gripper.set_vel = vel
            return True

        elif command.cmd_type == "set_force":
            force = command.params.get("force")
            gripper.set_force = force
            return True

        elif command.cmd_type == "get_actual_vel":
            vel = gripper.actual_vel
            return vel

        elif command.cmd_type == "get_set_vel":
            vel = gripper.set_vel
            return vel

        elif command.cmd_type == "get_cur":
            return gripper.actual_cur

        elif command.cmd_type == "get_max_force":
            return gripper.max_force

        elif command.cmd_type == "get_status":
            return gripper.status
        else:
            raise ValueError(f"Unknown command: {command.cmd_type}")

    def shutdown(self):
        """Close the gripper process"""
        self._terminate.set()
        self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()

    # Implementation of ParallelPositionGripper abstract methods

    def _send_command_and_wait_for_result(self, command: GripperCommand) -> None:
        """Send a command to the gripper process and wait for the result"""
        # Clear result queue - multiprocessing Queue doesn't have clear() method
        while not self._result_queue.empty():
            try:
                self._result_queue.get_nowait()
            except Exception as e:
                print(f"Error clearing result queue: {e}")
        self._cmd_queue.put(command)
        result = self._result_queue.get(timeout=1.0)
        if result["success"]:
            return result["result"]
        else:
            raise RuntimeError(f"Failed to execute command: {result['error']}")

    @property
    def speed(self) -> float:
        # send get_vel command to gripper process
        rel_speed = self._send_command_and_wait_for_result(GripperCommand(cmd_type="get_set_vel"))
        return rel_speed / 1000.0

    @speed.setter
    def speed(self, new_speed: float) -> None:
        new_speed = np.clip(new_speed, self.gripper_specs.min_speed, self.gripper_specs.max_speed)
        new_speed_mm = new_speed * 1000.0
        self._send_command_and_wait_for_result(GripperCommand(cmd_type="set_vel", params={"velocity": new_speed_mm}))
        return

    @property
    def max_grasp_force(self) -> float:
        force_relative = self._send_command_and_wait_for_result(GripperCommand(cmd_type="get_max_force"))
        return rescale_range(force_relative, 0, 100, self.gripper_specs.min_force, self.gripper_specs.max_force)

    @max_grasp_force.setter
    def max_grasp_force(self, new_force: float) -> None:
        new_force = np.clip(new_force, self.gripper_specs.min_force, self.gripper_specs.max_force)
        new_force_relative = rescale_range(
            new_force, self.gripper_specs.min_force, self.gripper_specs.max_force, 0, 100
        )
        self._send_command_and_wait_for_result(
            GripperCommand(cmd_type="set_force", params={"force": new_force_relative})
        )

    def get_current_width(self) -> float:
        """Get the current opening of the fingers in meters"""
        # Convert from mm to m
        with self._position.get_lock():
            position_mm_inverse = self._position.value
        return self.gripper_specs.max_width - position_mm_inverse / 1000.0

    def get_current_velocity(self) -> float:
        return self._send_command_and_wait_for_result(GripperCommand(cmd_type="get_actual_vel")) / 1000.0

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> AwaitableAction:
        """
        Move the fingers to the desired width between the fingers[m].
        Optionally provide a speed and/or force, that will be used from then on for all move commands.
        """
        width = np.clip(width, self.gripper_specs.min_width, self.gripper_specs.max_width)
        width_mm_gripper = (self.gripper_specs.max_width - width) * 1000.0

        if speed is not None:
            self.speed = speed
        if force is not None:
            self.max_grasp_force = force

        self._send_command_and_wait_for_result(
            GripperCommand(cmd_type="move_pos", params={"position": width_mm_gripper})
        )

        # Create awaitable action that checks if the gripper has reached the target position
        return AwaitableAction(lambda: not self.is_moving())

    def is_moving(self) -> bool:
        # check if speed is above minimum
        print(f"current velocity: {self.get_current_velocity()}")
        return abs(self.get_current_velocity()) > 1e-6

    def grasp_object(self) -> AwaitableAction:
        self._send_command_and_wait_for_result(GripperCommand(cmd_type="grip"))
        return AwaitableAction(self.is_moving)


if __name__ == "__main__":
    # Example usage
    gripper = SchunkGripperProcess("/dev/ttyUSB2,11")

    try:
        gripper.speed = 0.04
        target = 0.05
        position = gripper.get_current_width()
        print(f"Initial position: {position}")
        w = gripper.move(target)
        time.sleep(0.2)
        position = gripper.get_current_width()
        print(f"intermediate position: {position}")
        w.wait()
        position = gripper.get_current_width()
        print(f"final position: {position}")
        gripper.close().wait()
        gripper.move(0.01).wait()

        gripper.open().wait()
        gripper.move(0.01).wait()

    finally:
        print("done")
        gripper.shutdown()
