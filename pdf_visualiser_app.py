import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from distributions import DISTRIBUTIONS, DistributionSpec, scipy_distribution_info
from empirical_data import load_uploaded_sample_data, numeric_sample_columns
from formatting import format_mixture_components, format_params
from mixtures import MixtureComponent, calculate_mixture_density


st.set_page_config(
    page_title="Probability Density Visualiser",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Probability Density Visualiser")

with st.sidebar:
    # The sidebar controls what is shown and the shared x-axis range.
    st.header("Distributions")
    selected_distribution_ids = st.multiselect(
        "Show probability density functions",
        options=list(DISTRIBUTIONS.keys()),
        default=["norm"],
        format_func=lambda distribution_id: DISTRIBUTIONS[distribution_id].label,
    )

    st.header("Plot range")
    x_min = st.number_input("Minimum x", value=-6.0, step=0.5)
    x_max = st.number_input("Maximum x", value=6.0, step=0.5)
    n_points = st.slider("Resolution", min_value=100, max_value=10000, value=1000, step=100)

    st.header("Mixture")
    show_mixture = st.checkbox("Show mixture distribution", value=False)
    mixture_component_count = st.number_input(
        "Components",
        min_value=1,
        max_value=8,
        value=1,
        step=1,
        disabled=not show_mixture,
    )
    show_weighted_components = st.checkbox(
        "Show weighted components",
        value=False,
        disabled=not show_mixture,
    )

    st.header("Empirical data")
    show_empirical_data = st.checkbox("Show uploaded data histogram", value=False)
    uploaded_data_file = st.file_uploader(
        "Upload 1D sample data",
        type=["csv", "txt"],
        disabled=not show_empirical_data,
    )
    histogram_bin_count = st.slider(
        "Bins",
        min_value=5,
        max_value=200,
        value=30,
        step=5,
        disabled=not show_empirical_data,
    )
    histogram_opacity = st.slider(
        "Histogram opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.35,
        step=0.05,
        disabled=not show_empirical_data,
    )

    empirical_samples = np.array([], dtype=float)
    empirical_label = "Empirical data"
    if show_empirical_data and uploaded_data_file is not None:
        try:
            sample_data = load_uploaded_sample_data(uploaded_data_file)
            numeric_columns = numeric_sample_columns(sample_data)
            if not numeric_columns:
                st.error(
                    "The uploaded data must contain at least one column with "
                    "finite numeric values."
                )
            else:
                selected_sample_column = st.selectbox(
                    "Value column",
                    options=numeric_columns,
                    format_func=str,
                )
                selected_values = pd.to_numeric(
                    sample_data[selected_sample_column],
                    errors="coerce",
                ).dropna()
                empirical_samples = selected_values.to_numpy(dtype=float)
                empirical_samples = empirical_samples[np.isfinite(empirical_samples)]
                empirical_label = f"Empirical data: {selected_sample_column}"
                if empirical_samples.size == 0:
                    st.error("The selected column does not contain finite numeric values.")
                else:
                    st.caption(f"{empirical_samples.size:,} finite samples loaded.")
        except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
            st.error(str(error))

plot_panel, parameter_panel = st.columns([3.5, 1.2], gap="large")

with parameter_panel:
    st.subheader("Parameters")

    selected_distributions: list[tuple[str, DistributionSpec, dict[str, float]]] = []
    for distribution_id in selected_distribution_ids:
        # Look up the distribution specification and render its parameter
        # controls inside a labelled expander in the right-hand panel.
        spec = DISTRIBUTIONS[distribution_id]
        with st.expander(spec.label, expanded=True):
            params = spec.parameter_controls(distribution_id)
        selected_distributions.append((distribution_id, spec, params))

    if not selected_distribution_ids:
        st.caption("Select distributions in the sidebar to edit their parameters here.")

    mixture_components: list[MixtureComponent] = []
    if show_mixture:
        st.subheader("Mixture")
        distribution_ids = list(DISTRIBUTIONS.keys())
        for component_index in range(int(mixture_component_count)):
            default_distribution_id = "norm"
            default_distribution_position = distribution_ids.index(default_distribution_id)
            with st.expander(f"Component {component_index + 1}", expanded=True):
                distribution_id = st.selectbox(
                    "Distribution",
                    options=distribution_ids,
                    index=default_distribution_position,
                    format_func=lambda option: DISTRIBUTIONS[option].label,
                    key=f"mixture_{component_index}_distribution",
                )
                weight = st.number_input(
                    "Weight",
                    min_value=0.0,
                    value=1.0,
                    step=0.1,
                    key=f"mixture_{component_index}_weight",
                )
                spec = DISTRIBUTIONS[distribution_id]
                params = spec.parameter_controls(
                    f"mixture_{component_index}_{distribution_id}"
                )
            mixture_components.append(
                (component_index + 1, distribution_id, spec, weight, params)
            )

with plot_panel:
    # Stop early for invalid or incomplete input rather than trying to plot.
    if x_min >= x_max:
        st.error("Minimum x must be less than maximum x.")
    elif not selected_distributions and not show_mixture and empirical_samples.size == 0:
        st.info("Select at least one probability density function in the sidebar.")
    else:
        # Build the shared x grid. Every selected PDF is evaluated on this same
        # grid so all curves can be compared directly on one Plotly figure.
        x = np.linspace(x_min, x_max, n_points)
        fig = go.Figure()

        for distribution_id, spec, params in selected_distributions:
            # Evaluate and add one line trace per selected distribution.
            y = spec.pdf(x, params)
            params_text = format_params(params)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=spec.label,
                    customdata=[params_text] * len(x),
                    hovertemplate=(
                        "x=%{x:.3f}<br>"
                        "density=%{y:.5f}<br>"
                        "parameters: %{customdata}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        if empirical_samples.size > 0:
            counts, edges = np.histogram(
                empirical_samples,
                bins=int(histogram_bin_count),
                range=(x_min, x_max),
            )
            widths = np.diff(edges)
            bin_centres = edges[:-1] + widths / 2
            density = counts / (empirical_samples.size * widths)
            outside_range_count = int(
                np.count_nonzero(
                    (empirical_samples < x_min) | (empirical_samples > x_max)
                )
            )

            fig.add_trace(
                go.Bar(
                    x=bin_centres,
                    y=density,
                    width=widths,
                    name=empirical_label,
                    opacity=histogram_opacity,
                    marker_line_width=0,
                    customdata=np.column_stack([edges[:-1], edges[1:], counts]),
                    hovertemplate=(
                        "bin=[%{customdata[0]:.3f}, %{customdata[1]:.3f})<br>"
                        "count=%{customdata[2]:.0f}<br>"
                        "density=%{y:.5f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

            if outside_range_count > 0:
                st.caption(
                    f"{outside_range_count:,} of {empirical_samples.size:,} samples "
                    "are outside the current plot range."
                )

        if show_mixture:
            total_weight = sum(component[3] for component in mixture_components)
            if total_weight <= 0:
                st.warning("Mixture weights must sum to more than zero.")
            else:
                mixture = calculate_mixture_density(x, mixture_components)
                mixture_text = format_mixture_components(
                    mixture_components,
                    mixture.total_weight,
                )

                for component in mixture.components:
                    if show_weighted_components:
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=component.density,
                                mode="lines",
                                name=f"Component {component.index}",
                                line=dict(dash="dot"),
                                customdata=[
                                    (
                                        f"weight={component.normalized_weight:.5g}, "
                                        f"{format_params(component.params)}"
                                    )
                                ]
                                * len(x),
                                hovertemplate=(
                                    "x=%{x:.3f}<br>"
                                    "weighted density=%{y:.5f}<br>"
                                    "parameters: %{customdata}"
                                    "<extra>%{fullData.name}</extra>"
                                ),
                            )
                        )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mixture.density,
                        mode="lines",
                        name="Mixture",
                        line=dict(width=4),
                        customdata=[mixture_text] * len(x),
                        hovertemplate=(
                            "x=%{x:.3f}<br>"
                            "density=%{y:.5f}<br>"
                            "%{customdata}"
                            "<extra>%{fullData.name}</extra>"
                        ),
                    )
                )

        # Layout settings apply to the whole figure, not to individual traces.
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="Probability density",
            hovermode="x unified",
            legend_title_text="Trace",
            barmode="overlay",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=20, r=20, t=30, b=90),
        )

        st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Distribution Information")
show_distribution_info = st.checkbox(
    "Show information about the distributions involved",
    value=False,
)

if show_distribution_info:
    info_controls = st.columns(2)
    with info_controls[0]:
        include_selected_info = st.checkbox(
            "Selected PDFs",
            value=True,
            disabled=not selected_distributions,
        )
    with info_controls[1]:
        include_mixture_info = st.checkbox(
            "Mixture components",
            value=show_mixture,
            disabled=not show_mixture,
        )

    info_entries: list[tuple[str, str, dict[str, float]]] = []
    if include_selected_info:
        for distribution_id, spec, params in selected_distributions:
            info_entries.append((spec.label, distribution_id, params))
    if include_mixture_info:
        for index, distribution_id, spec, _weight, params in mixture_components:
            info_entries.append((f"Component {index}: {spec.label}", distribution_id, params))

    if not info_entries:
        st.caption("Select at least one distribution or mixture component to show SciPy information.")

    for title, distribution_id, params in info_entries:
        info = scipy_distribution_info(distribution_id, params)
        with st.expander(title, expanded=True):
            st.markdown(f"**SciPy object:** `scipy.stats.{info.scipy_name}`")
            st.write(info.description)
            st.markdown(f"**Current parameters:** `{format_params(params)}`")
            st.markdown(f"**Current support:** `{info.support}`")

            if info.shape_parameters:
                st.markdown("**Shape parameters**")
                st.table(
                    [
                        {
                            "Parameter": parameter.label,
                            "SciPy name": parameter.name,
                            "Domain": parameter.domain,
                            "Type": "integer" if parameter.integral else "real",
                        }
                        for parameter in info.shape_parameters
                    ]
                )
            else:
                st.caption("This distribution has no shape parameters.")

            st.markdown(f"[Open the SciPy reference]({info.reference_url})")
