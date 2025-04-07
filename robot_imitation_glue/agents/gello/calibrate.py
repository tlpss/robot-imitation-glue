from dataclasses import dataclass
from typing import Tuple
import numpy as np
import argparse
from robot_imitation_glue.agents.gello.dynamixel_driver import DynamixelDriver

@dataclass
class Args:
    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...] = (0, 0, 0, 0, 0, 0)
    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[float, ...] = (1, 1, -1, 1, 1, 1)
    """The joint angles that the GELLO is placed in at (in radians)."""

    gripper: bool = True
    """Whether or not the gripper is attached."""

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints


def get_config(args: Args) -> None:
    joint_ids = list(range(1, args.num_joints + 1))
    driver = DynamixelDriver(joint_ids, port=args.port, baudrate=57600)

    # assume that the joint state shouold be args.start_joints
    # find the offset, which is a multiple of np.pi/16 that minimizes the error between the current joint state and args.start_joints
    # this is done by brute force, we seach in a range of +/- 8pi

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = args.joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = args.start_joints[index]
        return np.abs(joint_i - start_i)

    for _ in range(10):
        driver.get_joints()  # warmup

    for _ in range(1):
        best_offsets = []
        curr_joints = driver.get_joints()
        for i in range(args.num_robot_joints):
            best_offset = 0
            best_error = 1e6
            for offset in np.linspace(
                -8 * np.pi, 8 * np.pi, 8 * 32 + 1
            ):  # intervals of pi/16
                error = get_error(offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            best_offsets.append(best_offset)
        print()
        print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
        print(
            "best offsets function of pi: ["
            + ", ".join([f"{int(np.round(x/(np.pi/16)))}*np.pi/16" for x in best_offsets])
            + " ]",
        )
        if args.gripper:
            print(
                "gripper open (degrees)       ",
                np.rad2deg(driver.get_joints()[-1]) - 0.2,
            )
            print(
                "gripper close (degrees)      ",
                np.rad2deg(driver.get_joints()[-1]) - 42,
            )

def main():
    parser = argparse.ArgumentParser(description="Calibrate GELLO robot joints")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", 
                        help="The port that GELLO is connected to")
    parser.add_argument("--start-joints", type=float, nargs="+", default=[-3.14, -1.57, 1.57, -1.57, -1.57, 0],
                        help="The joint angles that the GELLO is placed in at (in radians)")
    parser.add_argument("--joint-signs", type=float, nargs="+", default=[1, 1, -1, 1, 1, 1],
                        help="The signs for each joint (should be 1 or -1)")
    parser.add_argument("--gripper", action="store_true", default=True,
                        help="Whether or not the gripper is attached")
    parser.add_argument("--no-gripper", action="store_false", dest="gripper",
                        help="Disable gripper")
    
    args = parser.parse_args()
    
    # Convert to Args class
    calibration_args = Args(
        port=args.port,
        start_joints=tuple(args.start_joints),
        joint_signs=tuple(args.joint_signs),
        gripper=args.gripper
    )
    
    # Run calibration
    get_config(calibration_args)

if __name__ == "__main__":
    main()
