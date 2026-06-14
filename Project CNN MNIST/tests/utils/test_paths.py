from pathlib import Path
import tempfile

from src.utils.paths import (
    project_root,
    data_dir,
    configs_dir,
    artifacts_dir,
    logs_dir,
    checkpoints_dir,
    ensure_dir,
    resolve_path,
    relative_to_root,
    file_exists,
    ensure_parent_dir,
)

# pytest tests/utils/test_paths.py -v

def test_project_root_returns_path():
    root = project_root()

    assert isinstance(root, Path)
    assert root.exists()


def test_data_dir():
    path = data_dir()

    assert isinstance(path, Path)
    assert path.name == "data"


def test_configs_dir():
    path = configs_dir()

    assert isinstance(path, Path)
    assert path.name == "configs"


def test_artifacts_dir():
    path = artifacts_dir()

    assert isinstance(path, Path)
    assert path.name == "artifacts"


def test_logs_dir():
    path = logs_dir()

    assert isinstance(path, Path)
    assert path.name == "logs"


def test_checkpoints_dir():
    path = checkpoints_dir()

    assert isinstance(path, Path)
    assert path.name == "checkpoints"


def test_ensure_dir_creates_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "new_folder"

        assert not path.exists()

        ensure_dir(path)

        assert path.exists()
        assert path.is_dir()


def test_resolve_path():
    resolved = resolve_path(".")

    assert isinstance(resolved, Path)
    assert resolved.is_absolute()


def test_relative_to_root():
    path = relative_to_root("configs")

    assert isinstance(path, Path)
    assert path.name == "configs"


def test_file_exists_true():
    with tempfile.NamedTemporaryFile() as f:
        assert file_exists(f.name)


def test_file_exists_false():
    assert not file_exists("non_existing_file.txt")


def test_ensure_parent_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = (
            Path(tmpdir)
            / "a"
            / "b"
            / "c"
            / "test.txt"
        )

        ensure_parent_dir(file_path)

        assert file_path.parent.exists()
        assert file_path.parent.is_dir()


def test_ensure_parent_dir_returns_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "folder" / "file.txt"

        returned = ensure_parent_dir(file_path)

        assert returned == file_path