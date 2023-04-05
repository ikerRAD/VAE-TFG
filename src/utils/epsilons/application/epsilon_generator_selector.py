"""
Class for selecting different epsilon generators given the instance itself or an identification string
"""
from typing import Union, List

from src.utils.epsilons.infrastructure.always_same_epsilon_generator import (
    AlwaysSameEpsilonGenerator,
)
from src.utils.epsilons.infrastructure.always_same_epsilon_set_generator import (
    AlwaysSameEpsilonSetGenerator,
)
from src.utils.epsilons.infrastructure.same_epsilon_generator import SameEpsilonGenerator
from src.utils.epsilons.infrastructure.same_epsilon_set_generator import (
    SameEpsilonSetGenerator,
)
from src.utils.epsilons.infrastructure.only_one_each_time_epsilon_generator import (
    OnlyOneEachTimeEpsilonGenerator,
)
from src.utils.epsilons.infrastructure.all_random_each_time_epsilon_generator import (
    AllRandomEachTimeEpsilonGenerator,
)
from src.utils.epsilons.domain.epsilon_generator import EpsilonGenerator


class EpsilonGeneratorSelector:
    @staticmethod
    def possible_keys() -> List[str]:
        return [
            "always_same_epsilon",
            "always_same_epsilon_set",
            "same_epsilon",
            "same_epsilon_set",
            "only_one_each_time",
            "all_random_each_time",
        ]

    @staticmethod
    def select(epsilon_generator: Union[str, EpsilonGenerator]) -> EpsilonGenerator:
        if epsilon_generator == "always_same_epsilon":
            return AlwaysSameEpsilonGenerator()
        elif epsilon_generator == "always_same_epsilon_set":
            return AlwaysSameEpsilonSetGenerator()
        elif epsilon_generator == "same_epsilon":
            return SameEpsilonGenerator()
        elif epsilon_generator == "same_epsilon_set":
            return SameEpsilonSetGenerator()
        elif epsilon_generator == "only_one_each_time":
            return OnlyOneEachTimeEpsilonGenerator()
        elif epsilon_generator == "all_random_each_time":
            return AllRandomEachTimeEpsilonGenerator()
        else:
            return epsilon_generator
