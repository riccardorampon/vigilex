import pytest
from vigilex_core.migrate import migrate


def test_missing_migrations_fail_loudly(tmp_path):
    # wrong cwd / empty dir must not look like "nothing to apply"; raises before connecting
    with pytest.raises(FileNotFoundError, match="no .sql migrations"):
        migrate("postgresql://unused@localhost:1/x", tmp_path / "migrations")
