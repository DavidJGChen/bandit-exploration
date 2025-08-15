from dataclasses import dataclass

from numpy import float64, uint8

Reward = float64
Action = uint8


@dataclass(frozen=True)
class SampleOutput:
    outcome: Reward  # TODO: change this to a generic
    reward: Reward
    reward_obs: bool
