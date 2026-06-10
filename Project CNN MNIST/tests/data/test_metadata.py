from pathlib import Path

from src.data.metadata import MetadataManager

from src.data.schemas import (
    DatasetMetadataSchema,
    DatasetStatisticsSchema,
    ValidationReportSchema,
)

# pytest tests/data/test_metadata.py

# Fixtures

def create_manager(tmp_path):

    return MetadataManager(
        output_dir=tmp_path,
    )


# Metadata

def test_save_metadata(tmp_path):

    manager = create_manager(tmp_path)

    metadata = DatasetMetadataSchema(
        dataset_name="MNIST",
        version="1.0.0",
        source="torchvision",
        created_at="2026-06-10",
        total_samples=70000,
        num_classes=10,
        image_shape=(28, 28),
        description="MNIST dataset",
    )

    filepath = manager.save_metadata(
        metadata
    )

    assert filepath.exists()
    assert filepath.name == "metadata.json"


def test_load_metadata(tmp_path):

    manager = create_manager(tmp_path)

    metadata = DatasetMetadataSchema(
        dataset_name="MNIST",
        version="1.0.0",
        source="torchvision",
        created_at="2026-06-10",
        total_samples=70000,
        num_classes=10,
        image_shape=(28, 28),
        description="MNIST dataset",
    )

    manager.save_metadata(metadata)

    loaded = manager.load_metadata()

    assert loaded["dataset_name"] == "MNIST"
    assert loaded["num_classes"] == 10
    assert loaded["total_samples"] == 70000


# Statistics

def test_save_statistics(tmp_path):

    manager = create_manager(tmp_path)

    statistics = DatasetStatisticsSchema(
        total_samples=70000,
        train_samples=55000,
        valid_samples=5000,
        test_samples=10000,
        num_classes=10,
        pixel_min=0.0,
        pixel_max=1.0,
        pixel_mean=0.1307,
        pixel_std=0.3081,
    )

    filepath = manager.save_statistics(
        statistics
    )

    assert filepath.exists()
    assert filepath.name == "statistics.json"


def test_load_statistics(tmp_path):

    manager = create_manager(tmp_path)

    statistics = DatasetStatisticsSchema(
        total_samples=70000,
        train_samples=55000,
        valid_samples=5000,
        test_samples=10000,
        num_classes=10,
        pixel_min=0.0,
        pixel_max=1.0,
        pixel_mean=0.1307,
        pixel_std=0.3081,
    )

    manager.save_statistics(statistics)

    loaded = manager.load_statistics()

    assert loaded["total_samples"] == 70000
    assert loaded["pixel_mean"] == 0.1307
    assert loaded["pixel_std"] == 0.3081


# Validation Report

def test_save_validation_report(tmp_path):

    manager = create_manager(tmp_path)

    report = ValidationReportSchema(
        passed=True,
        missing_values=0,
        duplicate_rows=0,
        invalid_labels=0,
        total_records=70000,
        message="Validation passed",
    )

    filepath = manager.save_validation_report(
        report
    )

    assert filepath.exists()
    assert filepath.name == "validation_report.json"


def test_load_validation_report(tmp_path):

    manager = create_manager(tmp_path)

    report = ValidationReportSchema(
        passed=True,
        missing_values=0,
        duplicate_rows=0,
        invalid_labels=0,
        total_records=70000,
        message="Validation passed",
    )

    manager.save_validation_report(report)

    loaded = manager.load_validation_report()

    assert loaded["passed"] is True
    assert loaded["missing_values"] == 0
    assert loaded["invalid_labels"] == 0


# Internal Helpers

def test_save_json_helper(tmp_path):

    manager = create_manager(tmp_path)

    filepath = manager._save_json(
        {"hello": "world"},
        "test.json",
    )

    assert filepath.exists()
    assert filepath.name == "test.json"


def test_load_json_helper(tmp_path):

    manager = create_manager(tmp_path)

    manager._save_json(
        {"value": 123},
        "sample.json",
    )

    data = manager._load_json(
        "sample.json",
    )

    assert data["value"] == 123


# Summary

def test_summary(tmp_path, capsys):

    manager = create_manager(tmp_path)

    manager._save_json(
        {"a": 1},
        "metadata.json",
    )

    manager._save_json(
        {"b": 2},
        "statistics.json",
    )

    manager.summary()

    captured = capsys.readouterr()

    assert "Metadata Directory" in captured.out
    assert "metadata.json" in captured.out
    assert "statistics.json" in captured.out


# Output Directory

def test_output_directory_created(tmp_path):

    output_dir = tmp_path / "artifacts"

    manager = MetadataManager(
        output_dir=output_dir,
    )

    assert output_dir.exists()
    assert output_dir.is_dir()