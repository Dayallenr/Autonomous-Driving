"""
Tests for locating CARLA's bundled ``agents`` package.

The scan walks real filesystem roots, so the risk being guarded against is a
discovery routine that hangs or raises on an unusual machine — a permission-denied
system directory, a drive that disappears. Failing to find CARLA must be a quiet
None, never an exception, because every caller has a valid path for "not found".
"""
from __future__ import annotations

import pytest

from pathfinder.sim import carla_paths


def make_python_api(root, *, marker=True):
    """Build a directory that looks like CARLA's PythonAPI/carla."""
    api = root / "PythonAPI" / "carla"
    navigation = api / "agents" / "navigation"
    navigation.mkdir(parents=True)
    if marker:
        (navigation / "global_route_planner.py").write_text("# stub\n")
    return api


def test_carla_pythonapi_env_var_is_used(tmp_path, monkeypatch):
    api = make_python_api(tmp_path / "CARLA_0.9.15")
    monkeypatch.setenv("CARLA_PYTHONAPI", str(api))
    monkeypatch.delenv("CARLA_ROOT", raising=False)

    assert carla_paths.find_carla_python_api() == api


def test_carla_root_env_var_appends_pythonapi(tmp_path, monkeypatch):
    root = tmp_path / "CARLA_0.9.15"
    api = make_python_api(root)
    monkeypatch.delenv("CARLA_PYTHONAPI", raising=False)
    monkeypatch.setenv("CARLA_ROOT", str(root))

    assert carla_paths.find_carla_python_api() == api


def test_explicit_env_var_wins_over_carla_root(tmp_path, monkeypatch):
    preferred = make_python_api(tmp_path / "preferred")
    make_python_api(tmp_path / "fallback")
    monkeypatch.setenv("CARLA_PYTHONAPI", str(preferred))
    monkeypatch.setenv("CARLA_ROOT", str(tmp_path / "fallback"))

    assert carla_paths.find_carla_python_api() == preferred


def test_a_stale_env_var_does_not_resolve(tmp_path, monkeypatch):
    """A moved install must not silently resolve to a directory with no agents.

    Returning it anyway would defer the failure to an ImportError inside a
    benchmark run, far from the cause.
    """
    empty = make_python_api(tmp_path / "CARLA", marker=False)
    monkeypatch.setenv("CARLA_PYTHONAPI", str(empty))
    monkeypatch.delenv("CARLA_ROOT", raising=False)
    monkeypatch.setattr(carla_paths, "_candidate_roots", list)

    assert carla_paths.find_carla_python_api() is None


def test_missing_carla_returns_none_rather_than_raising(monkeypatch):
    monkeypatch.delenv("CARLA_PYTHONAPI", raising=False)
    monkeypatch.delenv("CARLA_ROOT", raising=False)
    monkeypatch.setattr(carla_paths, "_candidate_roots", list)
    monkeypatch.setattr(carla_paths.sys, "path", [])

    assert carla_paths.find_carla_python_api() is None


def test_scan_survives_unreadable_directories(tmp_path, monkeypatch):
    """A permission error on one candidate must not abort the whole scan."""
    class Hostile:
        name = "carla-hostile"

        def is_dir(self):
            raise PermissionError("denied")

    class Root:
        def exists(self):
            return True

        def iterdir(self):
            raise PermissionError("denied")

    monkeypatch.delenv("CARLA_PYTHONAPI", raising=False)
    monkeypatch.delenv("CARLA_ROOT", raising=False)
    monkeypatch.setattr(carla_paths, "_candidate_roots", lambda: [Root()])
    monkeypatch.setattr(carla_paths.sys, "path", [])

    assert carla_paths.find_carla_python_api() is None


def test_scan_finds_carla_under_a_candidate_root(tmp_path, monkeypatch):
    api = make_python_api(tmp_path / "CARLA_0.9.15")
    monkeypatch.delenv("CARLA_PYTHONAPI", raising=False)
    monkeypatch.delenv("CARLA_ROOT", raising=False)
    monkeypatch.setattr(carla_paths, "_candidate_roots", lambda: [tmp_path])
    monkeypatch.setattr(carla_paths.sys, "path", [])

    assert carla_paths.find_carla_python_api() == api


def test_scan_ignores_directories_without_carla_in_the_name(tmp_path, monkeypatch):
    """The name filter is what keeps the scan from walking whole drives."""
    make_python_api(tmp_path / "unrelated-project")
    monkeypatch.delenv("CARLA_PYTHONAPI", raising=False)
    monkeypatch.delenv("CARLA_ROOT", raising=False)
    monkeypatch.setattr(carla_paths, "_candidate_roots", lambda: [tmp_path])
    monkeypatch.setattr(carla_paths.sys, "path", [])

    assert carla_paths.find_carla_python_api() is None


def test_ensure_raises_with_actionable_hint_when_requested(monkeypatch):
    monkeypatch.delenv("CARLA_PYTHONAPI", raising=False)
    monkeypatch.delenv("CARLA_ROOT", raising=False)
    monkeypatch.setattr(carla_paths, "find_carla_python_api", lambda: None)

    with pytest.raises(ImportError, match="CARLA_ROOT"):
        carla_paths.ensure_agents_importable(raise_on_missing=True)


def test_ensure_inserts_the_located_directory_on_sys_path(tmp_path, monkeypatch):
    api = make_python_api(tmp_path / "CARLA")
    monkeypatch.setattr(carla_paths, "find_carla_python_api", lambda: api)
    fake_path = []
    monkeypatch.setattr(carla_paths.sys, "path", fake_path)

    assert carla_paths.ensure_agents_importable() == api
    assert fake_path[0] == str(api)
