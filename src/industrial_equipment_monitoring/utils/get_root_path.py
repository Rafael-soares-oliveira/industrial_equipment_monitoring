import os
import subprocess
from pathlib import Path


def get_git_root(path=None):
    """
    Retorna o diretório raiz do repositório Git.

    Args:
        path (str, optional): Caminho para verificar. Se None, usa o diretório atual.

    Returns:
        Path: Objeto Path do diretório raiz do Git
    Raises:
        RuntimeError: Se não estiver em um repositório Git
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
