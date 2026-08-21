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

import argparse
import json
from pathlib import Path

__all__ = ["render_writeup"]

#: The sentence that travels with the behaviour agent wherever it appears,
#: matching ``pathfinder/comparison_writeup.py``'s stance.
_NOT_PROJECT_WORK = (
    "CARLA's own behaviour agent — **not project work**. It reads the "
    "simulator's world state directly and appears only as a reference upper "
    "bound produced by the same routes, traffic, and scoring; nothing it "
    "does may be presented as this project's driving."
)


def _unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


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
    totals = report["summary"]["infraction_totals"]
    if not totals:
        return ["No scoreable or tracked infraction was committed."]
    lines = ["| Infraction | Count |", "|---|---|"]
    lines += [f"| {name} | {totals[name]} |" for name in sorted(totals)]
    return lines


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
    specs = report["episodes"]
    towns = _unique_in_order([spec["town"] for spec in specs])
    weathers = _unique_in_order([spec["weather"] for spec in specs])
    seeds = [spec["seed"] for spec in specs]
    behavior = report.get("behavior")

    lines = [
        f"# BehaviorAgent reference baseline — {report['backend']} backend",
        "",
        f"Generated {report['generated_at']} from [`{source}`]"
        f"({Path(source).name}), which records every number below along with "
        "the full episode specifications needed to re-run it. This file is "
        "rendered from that artifact by `pathfinder/reference_writeup.py`; "
        "edit the generator, not this file.",
        "",
        f"> **Scope: {report['scope']}.** {report['scope_note']}",
        ">",
        f"> {_NOT_PROJECT_WORK}",
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
        "## Suite",
        "",
        f"- Episodes: {len(specs)}",
        f"- Towns: {', '.join(towns)}",
        f"- Weathers: {', '.join(weathers)}",
        f"- Seeds: {min(seeds)}–{max(seeds)}",
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
    parser = argparse.ArgumentParser(
        description="Regenerate the Markdown write-up for a reference-baseline report."
    )
    parser.add_argument(
        "report", type=Path, help="path to a reference_baseline JSON report"
    )
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("kind") != "reference_baseline":
        raise SystemExit(f"{args.report} is not a reference_baseline report")
    output = args.report.with_suffix(".md")
    # See the note in ablation.py: the rendered write-up is not ASCII, and
    # write_text without an explicit encoding uses cp1252 on Windows.
    output.write_text(render_writeup(report, source=str(args.report)), encoding="utf-8")
    print(f"write-up written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
