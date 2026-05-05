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
- Inspect density values and parameters through Plotly hover labels.

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

## Project Structure

```text
.
├── pdf_visualiser_app.py  # Streamlit UI and plotting workflow
├── distributions.py       # SciPy distribution registry and parameter controls
├── formatting.py          # Hover label and mixture formatting helpers
├── requirements.txt       # Python dependencies
└── README.md
```

## Development Notes

The distribution list is generated from SciPy's continuous distribution names in
`scipy.stats`. Each distribution receives Streamlit controls for its shape
parameters plus `loc` and `scale`.
