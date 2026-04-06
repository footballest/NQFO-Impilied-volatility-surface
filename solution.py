"""Entry point for the final NQFO submission build."""

from __future__ import annotations

from pathlib import Path

from src.nqfo.io import (
    assert_valid_submission,
    build_submission_template,
    finalize_submission,
    load_input_frames,
    resolve_input_paths,
    write_submission,
)
from src.nqfo.pipeline import FINAL_MODEL_NAME, run_locked_pipeline


OUTPUT_FILENAME = "submission.csv"
MIN_PREDICTION = 5.0


def main() -> None:
    """Run the locked final pipeline and write the competition submission CSV."""
    project_root = Path(__file__).resolve().parent
    input_paths = resolve_input_paths(project_root)

    train_df, test_df, sample_submission_df = load_input_frames(project_root)

    prediction_df = run_locked_pipeline(
        train_df=train_df,
        test_df=test_df,
        min_prediction=MIN_PREDICTION,
        verbose=True,
    )

    submission_template = build_submission_template(
        test_df=test_df,
        sample_submission_df=sample_submission_df,
    )
    submission_df = finalize_submission(
        submission_template=submission_template,
        prediction_df=prediction_df,
    )

    assert_valid_submission(
        submission_df=submission_df,
        sample_submission_df=sample_submission_df,
        min_prediction=MIN_PREDICTION,
    )

    output_path = project_root / OUTPUT_FILENAME
    write_submission(submission_df, output_path)

    print("Final model: locked hybrid IV completion pipeline")
    print(f"Train file: {input_paths.train}")
    print(f"Test file: {input_paths.test}")
    print(f"Sample submission: {input_paths.sample_submission}")
    print(f"Rows written: {len(submission_df)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
