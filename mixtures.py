from dataclasses import dataclass

import numpy as np

from distributions import DistributionSpec


MixtureComponent = tuple[int, str, DistributionSpec, float, dict[str, float]]


@dataclass(frozen=True)
class WeightedComponentDensity:
    index: int
    distribution_id: str
    spec: DistributionSpec
    normalized_weight: float
    params: dict[str, float]
    density: np.ndarray


@dataclass(frozen=True)
class MixtureDensity:
    total_weight: float
    components: list[WeightedComponentDensity]
    density: np.ndarray


def calculate_mixture_density(
    x: np.ndarray,
    components: list[MixtureComponent],
) -> MixtureDensity:
    total_weight = sum(component[3] for component in components)
    if total_weight <= 0:
        raise ValueError("Mixture weights must sum to more than zero.")

    mixture_density = np.zeros_like(x, dtype=float)
    weighted_components = []

    for index, distribution_id, spec, weight, params in components:
        normalized_weight = weight / total_weight
        component_density = normalized_weight * spec.pdf(x, params)
        mixture_density += component_density
        weighted_components.append(
            WeightedComponentDensity(
                index=index,
                distribution_id=distribution_id,
                spec=spec,
                normalized_weight=normalized_weight,
                params=params,
                density=component_density,
            )
        )

    return MixtureDensity(
        total_weight=total_weight,
        components=weighted_components,
        density=mixture_density,
    )
