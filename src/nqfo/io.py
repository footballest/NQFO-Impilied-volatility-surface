"""I/O helpers for the final NQFO submission pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd


ROOT_FILENAMES = {
    "train": "train.csv",
    "test": "test.csv",
    "sample_submission": "sample_submission.csv",
}


class InputPaths(NamedTuple):
    """Resolved input file locations for the final pipeline."""

    train: Path
    test: Path
    sample_submission: Path | None


def _resolve_required_csv(project_root: Path, filename: str) -> Path:
    """Resolve a required CSV from project root first, then from data/."""
    candidates = [
        project_root / filename,
        project_root / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Required input file '{filename}' not found in project root or data/."
    )


def _resolve_optional_csv(project_root: Path, filename: str) -> Path | None:
    """Resolve an optional CSV from project root first, then from data/."""
    candidates = [
        project_root / filename,
        project_root / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_input_paths(project_root: Path) -> InputPaths:
    """Resolve train/test paths and optional sample-submission path."""
    return InputPaths(
        train=_resolve_required_csv(project_root, ROOT_FILENAMES["train"]),
        test=_resolve_required_csv(project_root, ROOT_FILENAMES["test"]),
        sample_submission=_resolve_optional_csv(
            project_root,
            ROOT_FILENAMES["sample_submission"],
        ),
    )


def load_input_frames(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load train, test, and optional sample submission data."""
    paths = resolve_input_paths(project_root)

    train = pd.read_csv(paths.train, parse_dates=["date"])
    test = pd.read_csv(paths.test, parse_dates=["date"])
    sample_submission = (
        pd.read_csv(paths.sample_submission)
        if paths.sample_submission is not None
        else None
    )
    return train, test, sample_submission


def build_submission_template(
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Build the final submission template.

    If sample_submission.csv is available, use its row order exactly.
    Otherwise, derive the template from missing test-row ids in sorted order.
    """
    if sample_submission_df is not None:
        required_cols = ["row_id"]
        missing_cols = [
            col for col in required_cols if col not in sample_submission_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"sample_submission.csv is missing required columns: {missing_cols}"
            )
        return sample_submission_df[["row_id"]].copy()

    template = (
        test_df.loc[test_df["iv_observed"].isna(), ["row_id"]]
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    return template


def finalize_submission(
    submission_template: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge predictions onto the submission template and return exact competition schema.
    """
    submission = submission_template.merge(
        prediction_df[["row_id", "iv_predicted"]],
        on="row_id",
        how="left",
    )
    return submission[["row_id", "iv_predicted"]]


def assert_valid_submission(
    submission_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame | None,
    min_prediction: float = 5.0,
) -> None:
    """Run final integrity checks on the submission dataframe."""
    expected_cols = ["row_id", "iv_predicted"]
    if submission_df.columns.tolist() != expected_cols:
        raise ValueError(
            f"Submission columns must be {expected_cols}, got {submission_df.columns.tolist()}."
        )

    if not submission_df["row_id"].is_unique:
        raise ValueError("Submission row_id values are not unique.")

    if submission_df["iv_predicted"].isna().any():
        raise ValueError("Submission contains missing predictions.")

    if (submission_df["iv_predicted"] < min_prediction).any():
        raise ValueError(
            f"Submission contains predictions below the minimum floor {min_prediction}."
        )

    if sample_submission_df is not None:
        sample_ids = sample_submission_df["row_id"].tolist()
        submission_ids = submission_df["row_id"].tolist()

        if len(sample_ids) != len(submission_ids):
            raise ValueError(
                f"Submission row count {len(submission_ids)} does not match sample submission row count {len(sample_ids)}."
            )

        if submission_ids != sample_ids:
            raise ValueError(
                "Submission row_id order does not exactly match sample_submission.csv."
            )


def write_submission(submission_df: pd.DataFrame, output_path: Path) -> None:
    """Write the final submission CSV."""
    submission_df.to_csv(output_path, index=False)
