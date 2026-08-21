"""
Render a policy-comparison report artifact into its Markdown write-up.

Generated from the report JSON rather than written by hand for the same
reason as the ablation's and the DAgger run's write-ups: issue #21 puts hard
requirements on it — the student's observation-boundary sentence, the
not-project-work labelling on the behaviour agent, the weak-floor caveat on
``pure_pursuit``, and traceability to the producing run — and a generated
document keeps those properties under test instead of under memory.

    python -m pathfinder.comparison_writeup results/comparison/carla_report.json

regenerates the sibling ``.md`` from an existing report. The comparison CLI
calls :func:`render_writeup` itself after every run, so normally the two
artifacts land together.
"""
from __future__ import annotations

from pathfinder.reporting import (
    NOT_PROJECT_WORK,
    generated_from,
    infraction_table,
    regenerate_writeup_main,
    scope_banner,
    suite_section,
)

__all__ = ["render_writeup"]

_ROLE_TITLES = {
    "floor": "Floor",
    "student": "Student",
    "reference_ceiling": "Reference ceiling",
}


def _column_heading(arm: dict) -> str:
    return f"{_ROLE_TITLES[arm['role']]} (`{arm['model_version']}`)"


def _floor_bullet(arm: dict) -> str:
    bullet = (
        f"- **{_ROLE_TITLES['floor']}** — `{arm['model_version']}`, the project's "
        "geometric controller, reading privileged state: exact cross-track "
        "error, heading error, path curvature, obstacle ranges, and "
        "traffic-light state straight from the simulator. It is a weak floor "
        "and a low bar: the perception ablation measured this controller — "
        "not perception — as the binding constraint on driving score"
    )
    if not arm.get("skipped"):
        shortfall = 100.0 - arm["summary"]["driving_score"]
        bullet += (
            f", and in this run its own shortfall from a perfect 100 is "
            f"{shortfall:.2f} points"
        )
    return bullet + ". A student that beats it has cleared a low bar, not demonstrated strong driving."


def _student_bullet(arm: dict, boundary: str) -> str:
    weights = f" (weights `{arm['weights']}`)" if arm.get("weights") else ""
    return (
        f"- **{_ROLE_TITLES['student']}** — `{arm['model_version']}`{weights}: "
        f"drives from {boundary}. This is a strictly harder task than either "
        "privileged baseline's — the baselines are handed the state the "
        "student must infer from pixels."
    )


def _reference_bullet(arm: dict) -> str:
    bullet = f"- **{_ROLE_TITLES['reference_ceiling']}** — `{arm['model_version']}`: {NOT_PROJECT_WORK}"
    if arm.get("skipped"):
        bullet += f" It did not run in this suite: {arm['skip_reason']}."
    return bullet


def _results_table(arms: list[dict]) -> list[str]:
    rows = [
        ("Driving score", "driving_score"),
        ("Route completion", "route_completion"),
        ("Infraction penalty", "infraction_penalty"),
        ("Collisions per km", "collisions_per_km"),
        ("Failures", "failures"),
    ]
    lines = [
        "| Metric | " + " | ".join(_column_heading(arm) for arm in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    for label, key in rows:
        cells = [
            "not run" if arm.get("skipped") else str(arm["summary"][key])
            for arm in arms
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(
        "| Mean policy latency (ms) | "
        + " | ".join(
            "not run" if arm.get("skipped") else str(arm["mean_policy_latency_ms"])
            for arm in arms
        )
        + " |"
    )
    return lines


def _infraction_table(arms: list[dict]) -> list[str]:
    return infraction_table(
        [
            (
                _column_heading(arm),
                None if arm.get("skipped") else arm["summary"]["infraction_totals"],
            )
            for arm in arms
        ],
        empty_message="No arm committed a scoreable or tracked infraction.",
    )


def _per_episode_table(report: dict) -> list[str]:
    arms = report["arms"]
    score_by_arm = {
        arm["role"]: {row["episode_id"]: row for row in arm.get("episodes", [])}
        for arm in arms
    }
    failed = set(report["failed_episodes"])
    lines = [
        "| Episode | Town | Weather | Seed | "
        + " | ".join(_column_heading(arm) for arm in arms)
        + " | Flag |",
        "|---|---|---|---|" + "---|" * (len(arms) + 1),
    ]
    for spec in report["episodes"]:
        episode_id = spec["episode_id"]
        cells = []
        flags = []
        for arm in arms:
            row = score_by_arm[arm["role"]].get(episode_id)
            if row is None:
                cells.append("not run")
                continue
            cells.append(str(row["driving_score"]))
            if episode_id in failed and row["status"] != "completed":
                flags.append(f"{arm['role']} {row['status']}")
        lines.append(
            f"| {episode_id} | {spec['town']} | {spec['weather']} | {spec['seed']} | "
            + " | ".join(cells)
            + f" | {', '.join(flags)} |"
        )
    return lines


def render_writeup(report: dict, *, source: str) -> str:
    """Render one policy-comparison report dict as a self-labelling Markdown
    write-up.

    Args:
        report: A ``run_comparison`` report, exactly as serialised to JSON.
        source: Path of the report artifact, as the write-up should cite it —
            the traceability link from every number back to the run.
    """
    arms = report["arms"]
    by_role = {arm["role"]: arm for arm in arms}

    lines = [
        f"# Policy comparison — {report['backend']} backend",
        "",
        generated_from(
            report, source=source, generator="pathfinder/comparison_writeup.py"
        ),
        "",
        scope_banner(report),
        "",
        "## The three columns",
        "",
        "All columns drive identical seeded episodes — same routes, traffic, "
        "weather, and scoring — and differ only in Policy. They do **not** "
        "observe the same world:",
        "",
        _floor_bullet(by_role["floor"]),
        _student_bullet(by_role["student"], report["student_observation_boundary"]),
        _reference_bullet(by_role["reference_ceiling"]),
        "",
        *suite_section(report["episodes"], episodes_label="Episodes per column"),
        "",
        "## Results",
        "",
        *_results_table(arms),
        "",
        "## Infraction breakdown",
        "",
        *_infraction_table(arms),
        "",
        "## Per-episode driving score",
        "",
        *_per_episode_table(report),
    ]

    failed = report["failed_episodes"]
    if failed:
        lines += [
            "",
            f"Episode(s) {', '.join(failed)} failed mid-run in at least one "
            "column; their scores measure the failure, not the Policy, and "
            "must be excluded from any comparison claim.",
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    return regenerate_writeup_main(
        argv,
        kind="policy_comparison",
        render_writeup=render_writeup,
        description="Regenerate the Markdown write-up for a policy-comparison report.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
