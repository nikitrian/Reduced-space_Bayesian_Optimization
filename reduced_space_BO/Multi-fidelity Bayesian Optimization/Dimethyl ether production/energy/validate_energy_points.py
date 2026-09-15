#!/usr/bin/env python
"""
Utility to evaluate the Aspen energy models on explicitly provided inputs.

Example usage:
    python validate_energy_points.py --norm-json "[0.77, ..., 0.0]"
    python validate_energy_points.py --csv candidates.csv --csv-kind norm --output results.csv

The script converts any normalised inputs back to the real engineering ranges,
optionally calls the ANN surrogate and the Aspen/HYSYS high-fidelity model,
and prints (and optionally saves) the results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

# Ensure relative assets in BO_energy.py are resolved from the energy directory.
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

from BO_energy import (
    SPACE,
    SPACE_COLS,
    energy_ann_lo,
    energy_sim_hi,
    real_to_unit01_np,
    unnormalize_x,
)

D = len(SPACE)
CLIP_TOL = 1e-8


def _extend_rows(container: List[np.ndarray], values: np.ndarray, origin: str) -> None:
    """Ensure `values` has shape (*, D) and append each row into container."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != D:
            raise ValueError(f"{origin} supplied {arr.size} values but {D} are required.")
        container.append(arr)
        return
    if arr.ndim != 2 or arr.shape[1] != D:
        raise ValueError(f"{origin} must have shape (*, {D}); received {arr.shape}.")
    for row in arr:
        container.append(np.asarray(row, dtype=np.float64))


def _frame_to_array(frame: pd.DataFrame, origin: str) -> np.ndarray:
    """Extract columns from a DataFrame in the proper order."""
    if set(SPACE_COLS).issubset(frame.columns):
        return frame[SPACE_COLS].to_numpy(dtype=np.float64)
    if frame.shape[1] != D:
        raise ValueError(
            f"{origin} must either include columns {SPACE_COLS} or exactly {D} unnamed columns."
        )
    return frame.to_numpy(dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    metavar_cols = tuple(SPACE_COLS)
    parser = argparse.ArgumentParser(
        description="Evaluate energy model responses for explicit candidate points."
    )
    parser.add_argument(
        "--norm",
        dest="norm_rows",
        action="append",
        nargs=D,
        type=float,
        metavar=metavar_cols,
        help="Normalised [0,1] decision variables (11 numbers per occurrence).",
    )
    parser.add_argument(
        "--norm-json",
        dest="norm_json",
        action="append",
        help="Normalised point(s) encoded as JSON; accepts either a list of 11 floats or a list of lists.",
    )
    parser.add_argument(
        "--real",
        dest="real_rows",
        action="append",
        nargs=D,
        type=float,
        metavar=metavar_cols,
        help="Real-domain decision variables; converted to [0,1] before evaluation.",
    )
    parser.add_argument(
        "--real-json",
        dest="real_json",
        action="append",
        help="Real-domain point(s) encoded as JSON; accepts either a list of 11 floats or a list of lists.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a CSV file containing candidate points (normalised by default).",
    )
    parser.add_argument(
        "--csv-kind",
        choices=("norm", "real"),
        default="norm",
        help="Interpret the CSV values as either normalised ('norm') or real-domain ('real').",
    )
    parser.add_argument(
        "--skip-hi",
        action="store_true",
        help="Skip the Aspen / high-fidelity model evaluation.",
    )
    parser.add_argument(
        "--skip-lo",
        action="store_true",
        help="Skip the ANN surrogate evaluation.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="Number of decimal places to display when printing tables (default: 6).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional CSV file to store the combined results.",
    )
    parser.add_argument(
        "--show-norm",
        action="store_true",
        help="Print an additional table with the normalised inputs.",
    )
    args = parser.parse_args()
    return args, parser


def _parse_json_points(raw_list: List[str], origin: str) -> List[np.ndarray]:
    parsed: List[np.ndarray] = []
    for entry in raw_list:
        try:
            payload = json.loads(entry)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{origin} string is not valid JSON: {exc}") from exc
        _extend_rows(parsed, payload, origin)
    return parsed


def _collect_normalised_points(args: argparse.Namespace, parser: argparse.ArgumentParser) -> np.ndarray:
    rows: List[np.ndarray] = []
    if args.norm_rows:
        for idx, row in enumerate(args.norm_rows, start=1):
            _extend_rows(rows, row, f"--norm occurrence #{idx}")
    if args.norm_json:
        rows.extend(_parse_json_points(args.norm_json, "--norm-json"))
    if args.real_rows:
        for idx, row in enumerate(args.real_rows, start=1):
            real = np.asarray(row, dtype=np.float64)
            norm = real_to_unit01_np(real, SPACE)
            _extend_rows(rows, norm, f"--real occurrence #{idx}")
    if args.real_json:
        real_points = _parse_json_points(args.real_json, "--real-json")
        for idx, real in enumerate(real_points, start=1):
            norm = real_to_unit01_np(real, SPACE)
            _extend_rows(rows, norm, f"--real-json entry #{idx}")

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            parser.error(f"CSV file not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        arr = _frame_to_array(frame, f"CSV '{csv_path}'")
        if args.csv_kind == "real":
            arr = real_to_unit01_np(arr, SPACE)
        _extend_rows(rows, arr, f"CSV '{csv_path}' ({args.csv_kind})")

    if not rows:
        parser.error("No candidate points supplied. Use --norm/--real/--csv options.")

    norm_arr = np.vstack(rows)
    if np.isnan(norm_arr).any():
        parser.error("Encountered NaNs in the provided inputs.")

    below = norm_arr < 0.0
    above = norm_arr > 1.0
    if np.any((norm_arr < -CLIP_TOL) | (norm_arr > 1.0 + CLIP_TOL)):
        parser.error(
            f"Normalised values fall outside [0,1] by more than {CLIP_TOL:g}. "
            f"Min={norm_arr.min():.6f}, max={norm_arr.max():.6f}"
        )
    if np.any(below | above):
        print(
            "[warn] Some normalised values were marginally outside [0,1]; "
            "they have been clipped to the boundary."
        )
        norm_arr = np.clip(norm_arr, 0.0, 1.0)
    return norm_arr


def main() -> None:
    args, parser = _parse_args()
    norm_arr = _collect_normalised_points(args, parser)
    tensor_norm = torch.from_numpy(norm_arr).to(dtype=torch.double)
    tensor_real = unnormalize_x(tensor_norm, SPACE)
    real_arr = tensor_real.cpu().numpy()

    lo_values = None
    hi_values = None
    if not args.skip_lo:
        lo_values = energy_ann_lo(tensor_norm).view(-1).cpu().numpy()
    if not args.skip_hi:
        hi_values = energy_sim_hi(tensor_norm).view(-1).cpu().numpy()

    records = []
    for idx, (norm_row, real_row) in enumerate(zip(norm_arr, real_arr)):
        record = {"point": idx}
        for name, norm_val in zip(SPACE_COLS, norm_row):
            record[f"norm_{name}"] = float(norm_val)
        for name, real_val in zip(SPACE_COLS, real_row):
            record[name] = float(real_val)
        if lo_values is not None:
            record["lo_energy"] = float(lo_values[idx])
        if hi_values is not None:
            record["hi_energy"] = float(hi_values[idx])
        if lo_values is not None and hi_values is not None:
            record["hi_minus_lo"] = float(hi_values[idx] - lo_values[idx])
        records.append(record)

    full_df = pd.DataFrame(records)

    float_format = f"{{:.{args.precision}f}}".format
    real_cols = ["point"] + SPACE_COLS
    print("Real-domain inputs:")
    print(full_df[real_cols].to_string(index=False, float_format=float_format))

    if args.show_norm:
        norm_cols = ["point"] + [f"norm_{name}" for name in SPACE_COLS]
        print("\nNormalised inputs:")
        print(full_df[norm_cols].to_string(index=False, float_format=float_format))

    output_cols = ["point"]
    if lo_values is not None:
        output_cols.append("lo_energy")
    if hi_values is not None:
        output_cols.append("hi_energy")
    if "hi_minus_lo" in full_df.columns:
        output_cols.append("hi_minus_lo")
    print("\nModel responses:")
    print(full_df[output_cols].to_string(index=False, float_format=float_format))

    if args.output:
        out_path = Path(args.output)
        full_df.to_csv(out_path, index=False)
        print(f"\nSaved full results to '{out_path}'.")


if __name__ == "__main__":
    main()
