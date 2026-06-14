from pathlib import Path
from datetime import datetime

from .export import (
    save_report_json,
    save_report_txt,
)


class ReportGenerator:
    """
    Generate experiment reports.
    """

    def __init__(
        self,
        report_dir: str | Path,
    ):
        self.report_dir = Path(report_dir)

        self.report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def generate_evaluation_report(
        self,
        metrics: dict,
        model_name: str,
    ):
        report = {
            "model": model_name,
            "timestamp":
                datetime.now().isoformat(),
            "metrics": metrics,
        }

        save_report_json(
            report,
            self.report_dir /
            "evaluation.json",
        )

        return report

    def generate_benchmark_report(
        self,
        benchmark: dict,
    ):
        save_report_json(
            benchmark,
            self.report_dir /
            "benchmark.json",
        )

        return benchmark

    def generate_classification_report(
        self,
        report_text: str,
    ):
        save_report_txt(
            report_text,
            self.report_dir /
            "classification_report.txt",
        )

    def generate_summary(
        self,
        metrics: dict,
        benchmark: dict,
    ):
        text = []

        text.append(
            "MODEL EVALUATION SUMMARY"
        )
        text.append("=" * 50)

        text.append("")
        text.append("Metrics:")

        for k, v in metrics.items():
            text.append(
                f"{k}: {v}"
            )

        text.append("")
        text.append("Benchmark:")

        for k, v in benchmark.items():
            text.append(
                f"{k}: {v}"
            )

        summary = "\n".join(text)

        save_report_txt(
            summary,
            self.report_dir /
            "summary.txt",
        )

        return summary