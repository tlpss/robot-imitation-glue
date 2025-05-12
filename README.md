# robot-imitation-glue

Framework to reduce engineering efforts in using imitation learning for robot manipulation.

Facilitates:
- Dataset Collection (Lerobot dataset format) through **teleoperation** (Gello and Spacemouse supported but you can easily add others)
- Modifying/filtering **Lerobot Datasets**
- Replaying datasets on real hardware (for debugging)
- **Evaluating trained policies** on real hardware and save rollout observations(supports Lerobot ACT/ DP, OpenVLA, Pi0 and Pi-Fast but you can easily add others)

Everything is visualized using Rerun.


The code is written modularly so that you can easily:

1) integrate a new hardware setup
2) evaluate new learning algorithms
3) integrate new teleoperation methods


## Some core design decisions and assumptions

- code to describe and train networks for policy is not part of this codebase, recommended flow is to collect dataset here, train in separate codebase and then evalute the trained policy using a webserver (to avoid dependency issues and maintain separation of concerns).

- policies and teleop devices are both abstracted as agents.

- we assume a single robot arm + gripper. Control interface is absolute end-effector poses + absolute gripper width.

- we use the Lerobot dataset format and tooling to store demonstrations and rollouts.

- Callbacks are provided to switch between different action formats for the dataset/ policies:
![](docs/action-flow.png)


## Examples

2 examples are available of how to integrate this codebase for your robot.
Both are powered using our airo-mono repo, but you can freely select a control stack for your hardware.

### UR5station

### UR3station


## Development

### installation

- clone this repo
- initialize and update submodules: `git submodule update --init`
- create the conda environment `conda env create -f environment.yaml`


