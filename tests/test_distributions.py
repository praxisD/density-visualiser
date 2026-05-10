import numpy as np
from scipy import stats

from distributions import (
    DISTRIBUTIONS,
    ScipyDistributionConfig,
    default_for_shape,
    display_label,
    display_parameter_name,
    finite_bound,
    format_bound,
    format_shape_domain,
    make_pdf,
    shape_names,
    shape_parameter_info,
    scipy_distribution_info,
    scipy_doc_summary,
    scipy_reference_url,
)


def test_display_label_uses_friendly_name():
    assert display_label("norm") == "Normal (norm)"


def test_display_label_falls_back_to_title_case():
    assert display_label("made_up_dist") == "Made Up Dist (made_up_dist)"


def test_display_parameter_name_uses_known_labels_and_fallback():
    assert display_parameter_name("dfn") == "Numerator degrees of freedom"
    assert display_parameter_name("shape_param") == "Shape Param"


def test_scipy_reference_url_points_to_distribution_documentation():
    assert scipy_reference_url("norm") == (
        "https://docs.scipy.org/doc/scipy/reference/generated/"
        "scipy.stats.norm.html"
    )


def test_scipy_doc_summary_reads_first_docstring_line():
    assert scipy_doc_summary("norm") == "A normal continuous random variable."


def test_format_bound_handles_finite_and_non_finite_values():
    assert format_bound(-2.5) == "-2.5"
    assert format_bound(float("-inf")) == "-inf"
    assert format_bound(float("inf")) == "inf"
    assert format_bound(float("nan")) == "undefined"


def test_format_shape_domain_uses_endpoint_inclusivity():
    assert format_shape_domain((0.0, float("inf")), (False, False)) == "(0, inf)"
    assert format_shape_domain((0.0, 1.0), (True, True)) == "[0, 1]"


def test_finite_bound_returns_none_for_non_finite_values():
    assert finite_bound(float("inf")) is None
    assert finite_bound(float("-inf")) is None
    assert finite_bound(float("nan")) is None
    assert finite_bound(-2.5) == -2.5


def test_default_for_shape_uses_override_and_respects_bounds():
    assert default_for_shape("df", 1.0, None, None) == 5.0
    assert default_for_shape("df", 1.0, 6.0, None) == 6.0
    assert default_for_shape("df", 1.0, None, 4.0) == 4.0
    assert default_for_shape("unknown", 1.5, None, None) == 1.5


def test_shape_names_reads_scipy_shape_metadata():
    assert shape_names("norm") == ()
    assert shape_names("beta") == ("a", "b")


def test_shape_parameter_info_reads_scipy_shape_metadata():
    beta_parameters = shape_parameter_info("beta")

    assert [parameter.name for parameter in beta_parameters] == ["a", "b"]
    assert [parameter.label for parameter in beta_parameters] == ["a", "b"]
    assert [parameter.domain for parameter in beta_parameters] == ["(0, inf)", "(0, inf)"]
    assert [parameter.integral for parameter in beta_parameters] == [False, False]


def test_scipy_distribution_info_includes_current_support():
    info = scipy_distribution_info("uniform", {"loc": 2.0, "scale": 3.0})

    assert info.scipy_name == "uniform"
    assert info.label == "Uniform (uniform)"
    assert info.description == "A uniform continuous random variable."
    assert info.shape_parameters == ()
    assert info.support == "[2, 5]"


def test_make_pdf_matches_scipy_distribution_pdf():
    pdf = make_pdf(ScipyDistributionConfig("norm", (), ()))
    x = np.array([-1.0, 0.0, 1.0])

    result = pdf(x, {"loc": 0.0, "scale": 1.0})

    np.testing.assert_allclose(result, stats.norm.pdf(x))


def test_distribution_registry_contains_working_normal_spec():
    spec = DISTRIBUTIONS["norm"]
    x = np.array([0.0])

    assert spec.label == "Normal (norm)"
    np.testing.assert_allclose(
        spec.pdf(x, {"loc": 0.0, "scale": 1.0}),
        stats.norm.pdf(x),
    )
