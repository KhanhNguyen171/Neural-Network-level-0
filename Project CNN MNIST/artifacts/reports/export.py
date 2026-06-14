from pathlib import Path
import json


def save_report_json(
    report: dict,
    path: str | Path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
        )


def save_report_txt(
    text: str,
    path: str | Path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text)