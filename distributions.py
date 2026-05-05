from dataclasses import dataclass
from math import isfinite
from typing import Callable

import numpy as np
from scipy import stats
from scipy.stats import _continuous_distns, _distr_params
import streamlit as st


@dataclass(frozen=True)
class DistributionSpec:
    # One entry in the distribution registry. Each distribution provides:
    # 1. a label for the UI,
    # 2. a function that creates its Streamlit controls,
    # 3. a function that evaluates the PDF for an array of x values.
    label: str
    parameter_controls: Callable[[str], dict[str, float]]
    pdf: Callable[[np.ndarray, dict[str, float]], np.ndarray]


@dataclass(frozen=True)
class ScipyDistributionConfig:
    scipy_name: str
    shape_names: tuple[str, ...]
    shape_defaults: tuple[float, ...]


DEFAULT_SHAPE_PARAMETERS = {
    name: params for name, params in _distr_params.distcont
}

FRIENDLY_LABELS = {
    "anglit": "Anglit",
    "argus": "ARGUS",
    "beta": "Beta",
    "betaprime": "Beta prime",
    "burr": "Burr",
    "burr12": "Burr XII",
    "cauchy": "Cauchy",
    "chi": "Chi",
    "chi2": "Chi-squared",
    "cosine": "Cosine",
    "crystalball": "Crystal Ball",
    "erlang": "Erlang",
    "expon": "Exponential",
    "exponnorm": "Exponentially modified normal",
    "exponpow": "Exponential power",
    "exponweib": "Exponentiated Weibull",
    "f": "F",
    "gamma": "Gamma",
    "gausshyper": "Gauss hypergeometric",
    "genextreme": "Generalized extreme value",
    "genlogistic": "Generalized logistic",
    "gennorm": "Generalized normal",
    "genpareto": "Generalized Pareto",
    "gompertz": "Gompertz",
    "halfcauchy": "Half-Cauchy",
    "halfgennorm": "Half-generalized normal",
    "halflogistic": "Half-logistic",
    "halfnorm": "Half-normal",
    "hypsecant": "Hyperbolic secant",
    "invgamma": "Inverse gamma",
    "invgauss": "Inverse Gaussian",
    "invweibull": "Inverse Weibull",
    "johnsonsb": "Johnson SB",
    "johnsonsu": "Johnson SU",
    "kstwobign": "Kolmogorov-Smirnov two-sided big N",
    "laplace": "Laplace",
    "laplace_asymmetric": "Asymmetric Laplace",
    "levy": "Levy",
    "levy_l": "Left-skewed Levy",
    "levy_stable": "Levy stable",
    "loggamma": "Log-gamma",
    "logistic": "Logistic",
    "loglaplace": "Log-Laplace",
    "lognorm": "Log-normal",
    "loguniform": "Log-uniform",
    "maxwell": "Maxwell",
    "moyal": "Moyal",
    "nakagami": "Nakagami",
    "ncf": "Noncentral F",
    "nct": "Noncentral t",
    "ncx2": "Noncentral chi-squared",
    "norm": "Normal",
    "norminvgauss": "Normal inverse Gaussian",
    "pareto": "Pareto",
    "powerlaw": "Power-function",
    "rayleigh": "Rayleigh",
    "rdist": "R",
    "recipinvgauss": "Reciprocal inverse Gaussian",
    "reciprocal": "Reciprocal",
    "rel_breitwigner": "Relativistic Breit-Wigner",
    "rice": "Rice",
    "semicircular": "Semicircular",
    "skewcauchy": "Skew-Cauchy",
    "skewnorm": "Skew-normal",
    "t": "Student's t",
    "triang": "Triangular",
    "truncexpon": "Truncated exponential",
    "truncnorm": "Truncated normal",
    "truncpareto": "Truncated Pareto",
    "truncweibull_min": "Truncated Weibull minimum",
    "tukeylambda": "Tukey-Lambda",
    "uniform": "Uniform",
    "vonmises": "Von Mises",
    "vonmises_line": "Von Mises line",
    "wald": "Wald",
    "weibull_max": "Weibull maximum",
    "weibull_min": "Weibull minimum",
    "wrapcauchy": "Wrapped Cauchy",
}


def display_label(scipy_name: str) -> str:
    label = FRIENDLY_LABELS.get(scipy_name, scipy_name.replace("_", " ").title())
    return f"{label} ({scipy_name})"


def display_parameter_name(parameter_name: str) -> str:
    labels = {
        "a": "a",
        "b": "b",
        "c": "c",
        "d": "d",
        "df": "Degrees of freedom",
        "dfd": "Denominator degrees of freedom",
        "dfn": "Numerator degrees of freedom",
        "k": "k",
        "lam": "Lambda",
        "loc": "Location",
        "nc": "Noncentrality",
        "scale": "Scale",
    }
    return labels.get(parameter_name, parameter_name.replace("_", " ").title())


def parameter_input(
    *,
    key: str,
    label: str,
    value: float,
    lower: float | None = None,
    upper: float | None = None,
    integral: bool = False,
) -> float:
    if integral:
        input_kwargs = {
            "value": int(round(value)),
            "step": 1,
            "key": key,
        }
        if lower is not None:
            input_kwargs["min_value"] = int(np.ceil(lower))
        if upper is not None:
            input_kwargs["max_value"] = int(np.floor(upper))
        return float(st.number_input(label, **input_kwargs))

    input_kwargs = {
        "value": float(value),
        "step": 0.1,
        "key": key,
    }
    if lower is not None:
        input_kwargs["min_value"] = float(lower)
    if upper is not None:
        input_kwargs["max_value"] = float(upper)
    return float(st.number_input(label, **input_kwargs))


def finite_bound(value: float) -> float | None:
    value = float(value)
    if not isfinite(value):
        return None
    return value


def default_for_shape(
    shape_name: str,
    shape_default: float,
    lower: float | None,
    upper: float | None,
) -> float:
    overrides = {
        "a": 2.0,
        "b": 3.0,
        "c": 2.0,
        "d": 3.0,
        "df": 5.0,
        "dfd": 10.0,
        "dfn": 5.0,
        "k": 3.0,
        "m": 2.0,
        "n": 5.0,
        "nc": 1.0,
    }
    value = float(overrides.get(shape_name, shape_default))
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    return value


def make_parameter_controls(config: ScipyDistributionConfig) -> Callable[[str], dict[str, float]]:
    distribution = getattr(stats, config.scipy_name)
    shape_info = distribution._shape_info()

    def controls(key_prefix: str) -> dict[str, float]:
        params: dict[str, float] = {}

        for index, info in enumerate(shape_info):
            lower = finite_bound(info.domain[0])
            upper = finite_bound(info.domain[1])
            shape_default = config.shape_defaults[index]
            value = default_for_shape(info.name, shape_default, lower, upper)
            params[info.name] = parameter_input(
                key=f"{key_prefix}_{info.name}",
                label=display_parameter_name(info.name),
                value=value,
                lower=lower,
                upper=upper,
                integral=info.integrality,
            )

        params["loc"] = parameter_input(
            key=f"{key_prefix}_loc",
            label=display_parameter_name("loc"),
            value=0.0,
        )
        params["scale"] = parameter_input(
            key=f"{key_prefix}_scale",
            label=display_parameter_name("scale"),
            value=1.0,
            lower=0.01,
        )
        return params

    return controls


def make_pdf(config: ScipyDistributionConfig) -> Callable[[np.ndarray, dict[str, float]], np.ndarray]:
    distribution = getattr(stats, config.scipy_name)

    def pdf(x: np.ndarray, params: dict[str, float]) -> np.ndarray:
        shape_args = [params[name] for name in config.shape_names]
        return distribution.pdf(
            x,
            *shape_args,
            loc=params["loc"],
            scale=params["scale"],
        )

    return pdf


def shape_names(scipy_name: str) -> tuple[str, ...]:
    shapes = getattr(stats, scipy_name).shapes
    if not shapes:
        return ()
    return tuple(shape.strip() for shape in shapes.split(","))


def build_distribution_spec(scipy_name: str) -> DistributionSpec:
    names = shape_names(scipy_name)
    defaults = tuple(float(value) for value in DEFAULT_SHAPE_PARAMETERS.get(scipy_name, ()))
    if len(defaults) != len(names):
        defaults = tuple(1.0 for _ in names)

    config = ScipyDistributionConfig(
        scipy_name=scipy_name,
        shape_names=names,
        shape_defaults=defaults,
    )
    return DistributionSpec(
        label=display_label(scipy_name),
        parameter_controls=make_parameter_controls(config),
        pdf=make_pdf(config),
    )


# SciPy's continuous distribution names are the PDFs this app can evaluate.
DISTRIBUTIONS: dict[str, DistributionSpec] = {
    scipy_name: build_distribution_spec(scipy_name)
    for scipy_name in _continuous_distns._distn_names
}
