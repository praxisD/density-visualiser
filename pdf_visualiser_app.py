import numpy as np
import plotly.graph_objects as go
import streamlit as st

from distributions import DISTRIBUTIONS, DistributionSpec


def format_params(params: dict[str, float]) -> str:
    # Used in hover text so the legend can stay compact.
    return ", ".join(f"{name}={value:g}" for name, value in params.items())


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

with plot_panel:
    # Stop early for invalid or incomplete input rather than trying to plot.
    if x_min >= x_max:
        st.error("Minimum x must be less than maximum x.")
    elif not selected_distributions:
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
                        "parameters=%{customdata}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        # Layout settings apply to the whole figure, not to individual traces.
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="Probability density",
            hovermode="x unified",
            legend_title_text="PDF",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.16,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=20, r=20, t=30, b=90),
        )

        st.plotly_chart(fig, use_container_width=True)

    # This note documents where the distribution list comes from.
    with st.expander("Distribution source"):
        st.markdown(
            """
            The distribution list is generated from SciPy's continuous
            distributions in `scipy.stats`. Each selected PDF gets controls for
            its shape parameters plus `loc` and `scale`.
            """
        )
