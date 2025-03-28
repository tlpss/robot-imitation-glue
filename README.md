# robot-imitation-glue

collect robot demonstrations, evaluate trained policies.




## Local Development

### Local installation

- clone this repo
- initialize and update submodules: `git submodule update --init`
- create the conda environment `conda env create -f environment.yaml`
- initialize the pre-commit hooks `pre-commit install`


### Running formatting, linting and testing
The makefile contains commands to make this convenient. Run using `make <command>`.