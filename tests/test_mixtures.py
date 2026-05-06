import numpy as np
import pytest

from distributions import DistributionSpec
from mixtures import calculate_mixture_density


def constant_spec(label: str, value: float) -> DistributionSpec:
    def pdf(x: np.ndarray, _params: dict[str, float]) -> np.ndarray:
        return np.full_like(x, value, dtype=float)

    return DistributionSpec(label, lambda _key: {}, pdf)


def test_calculate_mixture_density_normalizes_and_sums_components():
    x = np.array([0.0, 1.0, 2.0])
    first = constant_spec("First", 2.0)
    second = constant_spec("Second", 4.0)
    components = [
        (1, "first", first, 1.0, {"loc": 0.0, "scale": 1.0}),
        (2, "second", second, 3.0, {"loc": 0.0, "scale": 1.0}),
    ]

    result = calculate_mixture_density(x, components)

    assert result.total_weight == 4.0
    assert [component.normalized_weight for component in result.components] == [
        0.25,
        0.75,
    ]
    np.testing.assert_allclose(result.components[0].density, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(result.components[1].density, [3.0, 3.0, 3.0])
    np.testing.assert_allclose(result.density, [3.5, 3.5, 3.5])


def test_calculate_mixture_density_rejects_non_positive_total_weight():
    x = np.array([0.0])
    spec = constant_spec("Zero", 1.0)
    components = [(1, "zero", spec, 0.0, {"loc": 0.0, "scale": 1.0})]

    with pytest.raises(ValueError, match="sum to more than zero"):
        calculate_mixture_density(x, components)
