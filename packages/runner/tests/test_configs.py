"""Every configuration under ``configs/`` describes valid ``bnn`` runs of its own."""

from bnn.config import load_configs
from runner.paths import repository_root


def test_configurations_are_valid_and_their_runs_distinct():
    configs = repository_root() / "configs"
    folders = sorted({path.parent for path in configs.rglob("*.y*ml")})
    runs = [
        config.relative_directory
        for folder in folders
        for config in load_configs(folder)
    ]
    assert runs and len(set(runs)) == len(runs)
