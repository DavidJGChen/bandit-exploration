from dataclasses import dataclass
from typing import TypeAlias

from numpy import float64, uint8

Reward: TypeAlias = float64
Action: TypeAlias = uint8


@dataclass(frozen=True)
class SampleOutput:
    outcome: Reward  # TODO: change this to a generic
    reward: Reward
    reward_obs: bool
