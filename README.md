# Probability Density Visualiser

A Streamlit app for plotting and comparing probability density functions from
SciPy's continuous distributions. The app lets you select distributions, adjust
their parameters, set the x-axis range and resolution, and optionally build a
weighted mixture distribution.

## Features

- Plot one or more SciPy continuous probability density functions.
- Adjust distribution shape parameters, location, and scale from the UI.
- Control the plotting range and resolution.
- Build mixture distributions with configurable component weights.
- Optionally display weighted mixture components alongside the combined curve.
- Upload sample data and overlay a normalized histogram on the density plot.
- Tune the uploaded-data histogram bin count and opacity.
- Inspect density values and parameters through Plotly hover labels.
- Show distribution details for selected PDFs and mixture components, including
  links to the relevant SciPy documentation.

## Requirements

- Python 3.11.5
- Dependencies listed in `requirements.txt`

## Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

Start the Streamlit app from the project root:

```bash
streamlit run pdf_visualiser_app.py
```

Streamlit will print a local URL in the terminal, usually:

```text
http://localhost:8501
```

## Upload Empirical Data

Enable **Show uploaded data histogram** in the sidebar to overlay empirical
sample data on top of the PDF curves. The app accepts CSV files and plain text
files containing one-dimensional numeric sample data. For CSV files with
multiple numeric columns, choose the column to plot after upload.

Histograms are normalized as densities, so they can be compared directly with
the probability density functions on the same y-axis.

## Generate Sample Data

The repository includes a small utility that generates lognormal and Pareto
sample data for testing histogram overlays:

```bash
python sample_data/generate_lognormal_pareto.py
```

By default, this writes:

```text
sample_data/sample_lognormal_pareto.csv
```

You can change the output path, sample count, or random seed:

```bash
python sample_data/generate_lognormal_pareto.py --output sample_data/example.csv --samples 2000 --seed 7
```

## Project Structure

```text
.
├── pdf_visualiser_app.py             # Streamlit UI and plotting workflow
├── distributions.py                  # SciPy distribution registry, controls, and reference links
├── empirical_data.py                 # Uploaded sample data parsing helpers
├── formatting.py                     # Hover label and formatting helpers
├── mixtures.py                       # Mixture component construction and evaluation
├── sample_data/
│   └── generate_lognormal_pareto.py  # Sample CSV generator for histogram overlays
├── requirements.txt                  # Python dependencies
└── README.md
```

## Development Notes

The distribution list is generated from SciPy's continuous distribution names in
`scipy.stats`. Each distribution receives Streamlit controls for its shape
parameters plus `loc` and `scale`.

Distribution information shown in the app is resolved from SciPy metadata and
links to pages such as
`https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html`.
