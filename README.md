# bandit-exploration

## General todo:
- ~~Finish adding typing (mypy or ty)~~
- Add random seed to make results deterministic
- ~~Clean up images into a proper output file~~
- Add result saving
- Add basic testing
- ~~Add pre-commit~~
- Add command line args
- Add doc-strings
- Better integrate tqdm with multiprocessing. (For low trials, high T).

## Performance todo:
- Integrate better with Ray
- Figure out multiprocessing issue with IDS.

## Algorithm todo:
- Improve IDS action selection
- Get CVXPY to work?? -- (I'll need to reformulate in terms of quad_over_lin)
- Implement base IDS (non variance version)
- Add uncertainty intervals to regret plots
- Implement AI alignment bandit setting and algorithms for it.
