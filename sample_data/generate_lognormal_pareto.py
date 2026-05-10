import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


DEFAULT_SAMPLE_COUNT = 1000
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("sample_lognormal_pareto.csv")

LOGNORMAL_PARAMS = {
    "s": 0.5,
    "loc": 0.0,
    "scale": 1.0,
}

PARETO_PARAMS = {
    "b": 2.5,
    "loc": 0.0,
    "scale": 1.0,
}


def generate_samples(sample_count: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lognormal": stats.lognorm.rvs(
                **LOGNORMAL_PARAMS,
                size=sample_count,
                random_state=seed,
            ),
            "pareto": stats.pareto.rvs(
                **PARETO_PARAMS,
                size=sample_count,
                random_state=seed + 1,
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sample 1D data for the density visualiser."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV file to write.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of samples per distribution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed.",
    )
    args = parser.parse_args()

    samples = generate_samples(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
