"""
Data validation utilities.

This module validates dataset quality before the
data enters preprocessing or training pipelines.
"""

from pathlib import Path

import numpy as np

from .schemas import ValidationReportSchema


class DatasetValidator:
    """
    Validate MNIST dataset files.

    Expected files:

        X.npy
        y.npy
    """

    def __init__(self, x_path: str | Path, y_path: str | Path):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Load dataset files.
        """

        X = np.load(self.x_path)
        y = np.load(self.y_path)

        return X, y

    # ==========================================================
    # Validation Checks
    # ==========================================================

    def check_sample_count(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> int:
        """
        Ensure X and y contain the same number of samples.
        """

        return abs(len(X) - len(y))

    def check_missing_values(
        self,
        X: np.ndarray,
    ) -> int:
        """
        Count missing values.
        """

        return int(np.isnan(X).sum())

    def check_invalid_labels(
        self,
        y: np.ndarray,
    ) -> int:
        """
        MNIST labels must be in [0, 9].
        """

        invalid = np.logical_or(y < 0, y > 9)

        return int(np.sum(invalid))

    def check_duplicate_images(
        self,
        X: np.ndarray,
    ) -> int:
        """
        Count duplicated images.

        Images are flattened before duplicate search.
        """

        flattened = X.reshape(X.shape[0], -1)

        unique_count = np.unique(
            flattened,
            axis=0,
        ).shape[0]

        return int(len(X) - unique_count)

    def check_image_shape(
        self,
        X: np.ndarray,
    ) -> bool:
        """
        Verify image shape.

        Accepted:

            (N, 28, 28)
            (N, 1, 28, 28)
        """

        if X.ndim == 3:
            return X.shape[1:] == (28, 28)

        if X.ndim == 4:
            return X.shape[1:] == (1, 28, 28)

        return False

    def check_pixel_range(
        self,
        X: np.ndarray,
    ) -> bool:
        """
        Verify normalized pixel values.
        """

        return (
            float(X.min()) >= 0.0
            and float(X.max()) <= 1.0
        )

    # ==========================================================
    # Main Validation
    # ==========================================================

    def validate(self) -> ValidationReportSchema:
        """
        Run complete validation pipeline.
        """

        X, y = self._load()

        sample_mismatch = self.check_sample_count(X, y)

        missing_values = self.check_missing_values(X)

        invalid_labels = self.check_invalid_labels(y)

        duplicate_rows = self.check_duplicate_images(X)

        shape_ok = self.check_image_shape(X)

        range_ok = self.check_pixel_range(X)

        passed = all(
            [
                sample_mismatch == 0,
                missing_values == 0,
                invalid_labels == 0,
                shape_ok,
                range_ok,
            ]
        )

        message = (
            "Validation passed"
            if passed
            else "Validation failed"
        )

        return ValidationReportSchema(
            passed=passed,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            invalid_labels=invalid_labels,
            total_records=len(X),
            message=message,
        )

    # ==========================================================
    # Reporting
    # ==========================================================

    def summary(self) -> None:
        """
        Print validation summary.
        """

        report = self.validate()

        print("\nDataset Validation Report")
        print("-" * 40)

        print(f"Passed           : {report.passed}")
        print(f"Total Records    : {report.total_records}")
        print(f"Missing Values   : {report.missing_values}")
        print(f"Duplicate Images : {report.duplicate_rows}")
        print(f"Invalid Labels   : {report.invalid_labels}")
        print(f"Message          : {report.message}")