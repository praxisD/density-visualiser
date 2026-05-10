import io

import numpy as np
import pandas as pd


def columns_look_numeric(columns: pd.Index) -> bool:
    try:
        for column in columns:
            float(column)
    except (TypeError, ValueError):
        return False
    return True


def normalise_headerless_columns(data: pd.DataFrame) -> pd.DataFrame:
    column_count = len(data.columns)
    if column_count == 1:
        data.columns = ["value"]
    else:
        data.columns = [f"column_{index + 1}" for index in range(column_count)]
    return data


def numeric_sample_columns(data: pd.DataFrame) -> list[str]:
    columns = []
    for column in data.columns:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if np.isfinite(values.to_numpy(dtype=float)).any():
            columns.append(column)
    return columns


def load_uploaded_sample_data(uploaded_file) -> pd.DataFrame:
    raw_data = uploaded_file.getvalue()
    if not raw_data:
        raise ValueError("The uploaded file is empty.")

    if uploaded_file.name.lower().endswith(".txt"):
        text = raw_data.decode("utf-8")
        values = np.fromstring(text.replace(",", " "), sep=" ")
        if values.size == 0:
            raise ValueError("No numeric values were found in the text file.")
        return pd.DataFrame({"value": values})

    data = pd.read_csv(io.BytesIO(raw_data))
    if columns_look_numeric(data.columns):
        data = pd.read_csv(io.BytesIO(raw_data), header=None)
        data = normalise_headerless_columns(data)
    return data
