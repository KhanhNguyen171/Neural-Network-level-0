"""
Dataset metadata management.

Responsible for:

- Saving dataset metadata
- Saving dataset statistics
- Saving validation reports
- Loading metadata artifacts
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schemas import (
    DatasetMetadataSchema,
    DatasetStatisticsSchema,
    ValidationReportSchema,
)


class MetadataManager:
    """
    Manage metadata artifacts.
    """

    def __init__(self, output_dir: str | Path):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Internal Helpers

    def _save_json(
        self,
        data: dict,
        filename: str,
    ) -> Path:

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4,
            )

        return filepath

    def _load_json(
        self,
        filename: str,
    ) -> dict:

        filepath = self.output_dir / filename

        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    # Metadata

    def save_metadata(
        self,
        metadata: DatasetMetadataSchema,
    ) -> Path:

        return self._save_json(
            asdict(metadata),
            "metadata.json",
        )

    def load_metadata(self) -> dict:

        return self._load_json(
            "metadata.json",
        )

    # Statistics

    def save_statistics(
        self,
        statistics: DatasetStatisticsSchema,
    ) -> Path:

        return self._save_json(
            asdict(statistics),
            "statistics.json",
        )

    def load_statistics(self) -> dict:

        return self._load_json(
            "statistics.json",
        )

    # Validation Report

    def save_validation_report(
        self,
        report: ValidationReportSchema,
    ) -> Path:

        return self._save_json(
            asdict(report),
            "validation_report.json",
        )

    def load_validation_report(self) -> dict:

        return self._load_json(
            "validation_report.json",
        )

    # Summary

    def summary(self) -> None:

        print("\nMetadata Directory")
        print("-" * 40)

        for file in sorted(
            self.output_dir.glob("*.json")
        ):
            print(file.name)