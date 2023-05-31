"""
Class for selecting different epsilon generators given the instance itself or an identification string
"""
from enum import Enum
from typing import Union, List

from src.utils.epsilons.infrastructure.always_same_epsilon_generator import (
    AlwaysSameEpsilonGenerator,
)
from src.utils.epsilons.infrastructure.always_same_epsilon_set_generator import (
    AlwaysSameEpsilonSetGenerator,
)
from src.utils.epsilons.infrastructure.same_epsilon_generator import (
    SameEpsilonGenerator,
)
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
    class EpsilonGenerators(Enum):
        ALWAYS_SAME_EPSILON = "always_same_epsilon"
        ALWAYS_SAME_EPSILON_SET = "always_same_epsilon_set"
        SAME_EPSILON = "same_epsilon"
        SAME_EPSILON_SET = "same_epsilon_set"
        ONLY_ONE_EACH_TIME = "only_one_each_time"
        ALL_RANDOM_EACH_TIME = "all_random_each_time"

    @classmethod
    def possible_keys(cls) -> List[str]:
        return [elem.value for elem in cls.EpsilonGenerators]

    @classmethod
    def select(
        cls, epsilon_generator: Union[str, EpsilonGenerator]
    ) -> EpsilonGenerator:
        if epsilon_generator == cls.EpsilonGenerators.ALWAYS_SAME_EPSILON.value:
            return AlwaysSameEpsilonGenerator()

        if epsilon_generator == cls.EpsilonGenerators.ALWAYS_SAME_EPSILON_SET.value:
            return AlwaysSameEpsilonSetGenerator()

        if epsilon_generator == cls.EpsilonGenerators.SAME_EPSILON.value:
            return SameEpsilonGenerator()

        if epsilon_generator == cls.EpsilonGenerators.SAME_EPSILON_SET.value:
            return SameEpsilonSetGenerator()

        if epsilon_generator == cls.EpsilonGenerators.ONLY_ONE_EACH_TIME.value:
            return OnlyOneEachTimeEpsilonGenerator()

        if epsilon_generator == cls.EpsilonGenerators.ALL_RANDOM_EACH_TIME.value:
            return AllRandomEachTimeEpsilonGenerator()

        return epsilon_generator
