"""
Locating CARLA's bundled ``agents`` package.

The problem
-----------
CARLA ships two separate things. The pip wheel (``pip install carla``) provides
the ``carla`` module — client, world, actors, sensors. It does **not** provide
``agents``, which holds ``GlobalRoutePlanner`` and ``RoadOption``: the route
planning this project needs to produce real navigation commands for the
four-branch conditional planner.

``agents`` lives only inside the downloaded CARLA release, under
``PythonAPI/carla/``. The usual advice is to set ``PYTHONPATH`` by hand, which
works until the next shell, the next machine, or the next person cloning the
repo. Since the directory is discoverable, discover it.

Resolution order
----------------
1. ``CARLA_PYTHONAPI`` — an explicit path to the directory containing ``agents``.
2. ``CARLA_ROOT`` — the release root; ``PythonAPI/carla`` is appended.
3. ``agents`` already importable (someone set ``PYTHONPATH``, or it is installed).
4. A scan of conventional install locations per platform.

Every candidate is verified by checking that
``agents/navigation/global_route_planner.py`` actually exists, so a stale
environment variable pointing at a moved install fails here with a clear message
rather than as a confusing ``ImportError`` deep in a benchmark run.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "CARLA_HINT",
    "ensure_agents_importable",
    "find_carla_python_api",
    "route_planner_available",
]

#: Relative path that identifies a valid PythonAPI/carla directory.
_MARKER = Path("agents") / "navigation" / "global_route_planner.py"

CARLA_HINT = (
    "CARLA's 'agents' package was not found. It ships inside the CARLA release "
    "under PythonAPI/carla/ — the pip wheel does not include it.\n"
    "Point the repo at your install once:\n"
    "  Windows : setx CARLA_ROOT \"C:\\path\\to\\CARLA_0.9.15\"\n"
    "  Linux   : export CARLA_ROOT=/path/to/CARLA_0.9.15\n"
    "(open a new terminal after setx), or set CARLA_PYTHONAPI directly to the "
    "directory that contains 'agents'."
)


def _is_python_api_dir(candidate: Path) -> bool:
    try:
        return (candidate / _MARKER).is_file()
    except OSError:
        return False


def _candidate_roots() -> list[Path]:
    """Conventional CARLA install locations, most likely first."""
    roots: list[Path] = []
    home = Path.home()

    if sys.platform == "win32":
        drives = [Path(f"{letter}:/") for letter in "CDEFG" if Path(f"{letter}:/").exists()]
        for drive in drives:
            roots.extend([drive, drive / "Program Files", drive / "Games"])
        roots.extend([home, home / "Downloads", home / "Desktop"])
    else:
        roots.extend([Path("/opt"), Path("/usr/local"), home, home / "Downloads"])

    return [root for root in roots if root.exists()]


def find_carla_python_api() -> Path | None:
    """Locate the directory containing CARLA's ``agents`` package.

    Returns None when nothing is found; callers decide whether that is fatal.
    """
    explicit = os.environ.get("CARLA_PYTHONAPI")
    if explicit:
        candidate = Path(explicit).expanduser()
        if _is_python_api_dir(candidate):
            return candidate
        logger.warning("CARLA_PYTHONAPI=%s does not contain %s", explicit, _MARKER)

    root = os.environ.get("CARLA_ROOT")
    if root:
        candidate = Path(root).expanduser() / "PythonAPI" / "carla"
        if _is_python_api_dir(candidate):
            return candidate
        logger.warning("CARLA_ROOT=%s has no %s", root, Path("PythonAPI/carla") / _MARKER)

    # Already on sys.path — nothing to do.
    for entry in sys.path:
        if entry and _is_python_api_dir(Path(entry)):
            return Path(entry)

    # Shallow scan: CARLA unpacks into a directory whose name contains "carla",
    # so match that rather than walking entire drives, which on Windows can take
    # minutes and hit permission errors on system folders.
    for base in _candidate_roots():
        try:
            entries = list(base.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if not entry.is_dir() or "carla" not in entry.name.lower():
                continue
            for candidate in (entry / "PythonAPI" / "carla", entry):
                if _is_python_api_dir(candidate):
                    return candidate
    return None


def ensure_agents_importable(*, raise_on_missing: bool = False) -> Path | None:
    """Put CARLA's ``agents`` package on ``sys.path`` if it can be found.

    Args:
        raise_on_missing: Raise instead of returning None. Use where the caller
            genuinely cannot proceed, so the failure names the fix.

    Raises:
        ImportError: If ``raise_on_missing`` and the package cannot be located.
    """
    try:
        import agents.navigation.global_route_planner  # noqa: F401

        return None  # already importable
    except ImportError:
        pass

    location = find_carla_python_api()
    if location is None:
        if raise_on_missing:
            raise ImportError(CARLA_HINT)
        logger.warning("%s", CARLA_HINT)
        return None

    sys.path.insert(0, str(location))
    logger.info("added CARLA PythonAPI to sys.path: %s", location)
    return location


def route_planner_available() -> bool:
    """True when ``GlobalRoutePlanner`` can be imported, after discovery."""
    ensure_agents_importable()
    try:
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # noqa: F401

        return True
    except ImportError:
        return False
