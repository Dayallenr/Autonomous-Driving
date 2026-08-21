"""
Render a reference-baseline report artifact into its Markdown write-up.

Generated from the report JSON rather than written by hand for the same
reason as the ablation's and the comparison's write-ups: issue #16 puts hard
requirements on it — the not-project-work labelling, the scope of the
numbers, and the go / stop-and-reassess gate against the PurePursuit floor —
and a generated document keeps those properties under test instead of under
memory.

    python -m pathfinder.reference_writeup results/reference/carla_report.json

regenerates the sibling ``.md`` from an existing report. The reference-run
CLI calls :func:`render_writeup` itself after every run, so normally the two
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


def _results_table(report: dict) -> list[str]:
    summary = report["summary"]
    rows = [
        ("Driving score", summary["driving_score"]),
        ("Route completion", summary["route_completion"]),
        ("Infraction penalty", summary["infraction_penalty"]),
        ("Collisions per km", summary["collisions_per_km"]),
        ("Failures", summary["failures"]),
        ("Mean policy latency (ms)", report["mean_policy_latency_ms"]),
    ]
    lines = [
        f"| Metric | `{report['model_version']}` |",
        "|---|---|",
    ]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return lines


def _infraction_table(report: dict) -> list[str]:
    return infraction_table(
        [("Count", report["summary"]["infraction_totals"])],
        empty_message="No scoreable or tracked infraction was committed.",
    )


def _per_episode_table(report: dict) -> list[str]:
    failed = set(report["failed_episodes"])
    row_by_id = {row["episode_id"]: row for row in report["results"]}
    lines = [
        "| Episode | Town | Weather | Seed | Driving score | Flag |",
        "|---|---|---|---|---|---|",
    ]
    for spec in report["episodes"]:
        row = row_by_id[spec["episode_id"]]
        flag = row["status"] if spec["episode_id"] in failed else ""
        lines.append(
            f"| {spec['episode_id']} | {spec['town']} | {spec['weather']} | "
            f"{spec['seed']} | {row['driving_score']} | {flag} |"
        )
    return lines


def _gate_section(gate: dict) -> list[str]:
    if not gate.get("computed"):
        return [
            f"**Not computed** — {gate.get('reason', 'no reason recorded')}.",
        ]
    lines = [
        gate["definition"],
        "",
        # Cited as a path, not a relative link: the floor artifact lives in a
        # different directory than this write-up, so a sibling-style link
        # would dangle.
        f"- Floor: {gate['floor_driving_score']} — {gate['floor_policy']}, "
        f"from `{gate['floor_source']}`",
        f"- Reference: {gate['reference_driving_score']}",
        f"- Margin: {gate['margin']} (required: {gate['required_margin']})",
        "",
        f"**Verdict: {gate['verdict']}**",
    ]
    if gate.get("caveat"):
        lines += ["", f"Caveat: {gate['caveat']}."]
    return lines


def render_writeup(report: dict, *, source: str) -> str:
    """Render one reference-baseline report dict as a self-labelling Markdown
    write-up.

    Args:
        report: A ``run_reference`` report, exactly as serialised to JSON.
        source: Path of the report artifact, as the write-up should cite it —
            the traceability link from every number back to the run.
    """
    behavior = report.get("behavior")

    lines = [
        f"# BehaviorAgent reference baseline — {report['backend']} backend",
        "",
        generated_from(
            report, source=source, generator="pathfinder/reference_writeup.py"
        ),
        "",
        scope_banner(report),
        ">",
        f"> {NOT_PROJECT_WORK}",
        "",
        "## What this measures",
        "",
        f"`{report['model_version']}`"
        + (f" (behaviour preset `{behavior}`)" if behavior else "")
        + " driving the perception ablation's seeded suite. Its score is the "
        "reference ceiling for the Phase 3 comparison table and the "
        "teacher-quality measurement the DAgger training design depends on — "
        "an upper bound to cite alongside project results, never as one.",
        "",
        *suite_section(report["episodes"], episodes_label="Episodes"),
        "",
        "## Results",
        "",
        *_results_table(report),
        "",
        "## Infraction breakdown",
        "",
        *_infraction_table(report),
        "",
        "## Per-episode driving score",
        "",
        *_per_episode_table(report),
        "",
        "## Floor gate — go / stop-and-reassess (issue #16)",
        "",
        *_gate_section(report["floor_gate"]),
    ]

    failed = report["failed_episodes"]
    if failed:
        lines += [
            "",
            f"Episode(s) {', '.join(failed)} failed mid-run; their scores "
            "measure the failure, not the agent, and must be excluded from "
            "any ceiling claim.",
        ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    return regenerate_writeup_main(
        argv,
        kind="reference_baseline",
        render_writeup=render_writeup,
        description="Regenerate the Markdown write-up for a reference-baseline report.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
