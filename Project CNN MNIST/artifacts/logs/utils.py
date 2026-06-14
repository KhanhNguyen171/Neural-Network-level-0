from pathlib import Path
from datetime import datetime


def timestamp():
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def ensure_log_dir(
    path: str | Path,
):
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path