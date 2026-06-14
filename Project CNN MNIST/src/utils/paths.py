from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def project_root() -> Path:
    """
    Return project root directory.

    Assumes:
    project/
    ├── src/
    │   └── utils/
    │       └── paths.py
    """
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Return data directory."""
    return project_root() / "data"


def configs_dir() -> Path:
    """Return configs directory."""
    return project_root() / "configs"


def artifacts_dir() -> Path:
    """Return artifacts directory."""
    return project_root() / "artifacts"


def logs_dir() -> Path:
    """Return artifacts/logs directory."""
    return artifacts_dir() / "logs"


def checkpoints_dir() -> Path:
    """Return artifacts/checkpoints directory."""
    return artifacts_dir() / "checkpoints"


def ensure_dir(path: PathLike) -> Path:
    """
    Create directory if it does not exist.

    Returns
    -------
    Path
        Created directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: PathLike) -> Path:
    """
    Resolve path to absolute path.
    """
    return Path(path).expanduser().resolve()


def relative_to_root(path: PathLike) -> Path:
    """
    Convert project-relative path to absolute path.
    """
    return project_root() / Path(path)


def file_exists(path: PathLike) -> bool:
    """
    Check whether file exists.
    """
    return Path(path).exists()


def ensure_parent_dir(path: PathLike) -> Path:
    """
    Create parent directory for a file path.
    """
    path = Path(path)

    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)

    return path