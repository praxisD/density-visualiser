from distributions import DistributionSpec
from formatting import format_mixture_components, format_params


def test_format_params_uses_compact_numeric_formatting():
    assert format_params({"loc": 0.0, "scale": 1.23456}) == "loc=0, scale=1.23456"


def test_format_mixture_components_normalizes_weights():
    spec = DistributionSpec("Normal", lambda _key: {}, lambda x, params: x)
    components = [(1, "norm", spec, 2.0, {"loc": 0.0, "scale": 1.0})]

    assert format_mixture_components(components, 4.0) == (
        "1: 0.5 x Normal(loc=0, scale=1)"
    )
