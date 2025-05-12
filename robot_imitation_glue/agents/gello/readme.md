# Gello Teleop (for UR e-series)


original research project: https://wuphilipp.github.io/gello_site/

code based on :https://github.com/tlpss/gello_software/tree/main


Hardware instructions: see original repo.


## Usage

1. calibrate the teleop arm by holding it to a configuration and setting this configuration joint state manually in
`calibrate.py`. This will calculate offsets to add to the absolute encodings of the dynamixel motors, so that the joint angle output of the teleop agent can be applied directly to the UR robot.

2. take the calibrated joint offsets and add them to the Config for the agent: `gello_agent.py`


#TODO: extend instructions, provide image of setup.