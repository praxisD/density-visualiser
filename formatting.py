from mixtures import MixtureComponent


def format_params(params: dict[str, float]) -> str:
    # Used in hover text so the legend can stay compact.
    return ", ".join(f"{name}={value:g}" for name, value in params.items())


def format_mixture_components(
    components: list[MixtureComponent],
    total_weight: float,
) -> str:
    component_descriptions = []
    for index, _distribution_id, spec, weight, params in components:
        normalized_weight = weight / total_weight
        component_descriptions.append(
            f"{index}: {normalized_weight:.3g} x {spec.label}({format_params(params)})"
        )
    return "<br>".join(component_descriptions)
