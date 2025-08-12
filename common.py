from dataclasses import dataclass
from numpy import float64, int_

Reward = float64
Action = int_


@dataclass(frozen=True)
class SampleOutput:
    outcome: Reward  # TODO: change this to a generic
    reward: Reward
    reward_obs: bool
