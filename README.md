## About:

This is a (WIP) bandit simulator built in Python that started out as a course project, and eventually developed into a more ambitious effort.

The goal of this project is to provide a simple command-line interface to run easily customizable bandit simulations, with relatively fast, parallelizable (eventually distributed) computation.

The simulation results are saved in output in a way that is intended to facilitate follow-up analysis.

### Bandit settings and algorithm details:

A bandit simulation consists of two customizable parts:

1) The bandit environment (arms, hidden parameters, etc.)
2) The bandit algorithm (per-timestep logic, simulation state.)

Some algorithms and environments rely on certain Bayesian assumptions, so auxillary "Bayesian state" can be provided along with the bandit environment, for the algorithm to use.

## Running the simulator:

If running as a pre-built package, simply write:
```
bandit-sim <args>
```

If developing, please make sure a compatible version of uv is installed and run:
```
uv run bandit-sim <args>
```

## Running the notebooks:

Since these re-use some util functions in the bandit-sim module, make sure it is built with:
```
uv build
```
and then you should be able to import as necessary.


---
## TODO Section:

### General todo:
- Add result saving with pandas/polars
- Add basic testing
- Add **better** command line args
- Add config-file
- Add more doc-strings
- Better integrate tqdm with multiprocessing. (For low trials, high T).
- Update pyproject.toml (reference [this guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)).
- ~~Add pre-commit~~
- ~~Finish adding typing (mypy or ty)~~
- ~~Add random seed to make results deterministic~~
- ~~Clean up images into a proper output file~~

### Performance todo:
- Integrate better with Ray
- Figure out multiprocessing issue with IDS.

### Algorithm todo:
- Improve IDS action selection
- Get CVXPY to work?? -- (I'll need to reformulate in terms of quad_over_lin)
- Implement base IDS (non variance version)
- Figure out IDS choosing non-optimal actions suddenly (likely due to numerial precision errors?)
- ~~Add uncertainty intervals to regret plots~~
- ~~Implement AI alignment bandit setting and algorithms for it.~~
