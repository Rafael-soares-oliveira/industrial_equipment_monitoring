import os
import subprocess
from pathlib import Path


def get_git_root(path=None):
    """
    Returns the root git directory.

    Args:
        path (str, optional): Path to check. If None, use the actual directory.

    Returns:
        Path: Path of the root directory.
    Raises:
        RuntimeError: If it's not in a Git repository.
    """
    if path is None:
        path = os.getcwd()

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        raise RuntimeError(f"O caminho '{path}' não está em um repositório Git")
