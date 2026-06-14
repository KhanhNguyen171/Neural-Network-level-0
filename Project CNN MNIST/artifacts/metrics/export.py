from pathlib import Path
import json

import pandas as pd


def save_metrics_json(
    metrics: dict,
    path: str | Path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )


def save_metrics_csv(
    metrics: dict,
    path: str | Path,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.DataFrame([metrics])

    df.to_csv(
        path,
        index=False,
    )


def load_metrics(
    path: str | Path,
):
    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)