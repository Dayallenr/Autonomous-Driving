"""
Render an ablation report artifact into its Markdown write-up.

The write-up is generated from the report JSON rather than written by hand
because issue #10 puts hard requirements on it — the exact perception-boundary
sentence, an infraction breakdown, traceability to the producing run, and a
recorded (never acted-on) fine-tuning trigger — and a generated document keeps
those properties under test instead of under memory. Prose interpretation of a
specific run belongs in the README or the issue, quoting this file's numbers;
this document is the numbers, labelled with exactly what they may claim.

    python -m pathfinder.ablation_writeup results/ablation/carla_report.json

regenerates the sibling ``.md`` from an existing report. ``pathfinder.ablation``
calls :func:`render_writeup` itself after every run, so normally the two
artifacts land together.
"""
from __future__ import annotations

import statistics

from pathfinder.reporting import (
    SCOPE_DRIVING_QUALITY,
    generated_from,
    infraction_table,
    regenerate_writeup_main,
    scope_banner,
    suite_section,
)

__all__ = ["render_writeup"]


def _results_table(report: dict) -> list[str]:
    baseline, candidate = report["baseline"]["summary"], report["candidate"]["summary"]
    difference = report["difference"]
    rows = [
        ("Driving score", "driving_score", difference["driving_score"]),
        ("Route completion", "route_completion", difference["route_completion"]),
        ("Infraction penalty", "infraction_penalty", difference["infraction_penalty"]),
        ("Collisions per km", "collisions_per_km", None),
        ("Failures", "failures", None),
    ]
    lines = [
        f"| Metric | Baseline ({report['baseline']['perception']}) "
        f"| Candidate ({report['candidate']['perception']}) | Difference (B − C) |",
        "|---|---|---|---|",
    ]
    for label, key, delta in rows:
        lines.append(
            f"| {label} | {baseline[key]} | {candidate[key]} "
            f"| {'—' if delta is None else delta} |"
        )
    lines.append(
        "| Mean perception latency (ms) "
        f"| {report['baseline']['mean_perception_latency_ms']} "
        f"| {report['candidate']['mean_perception_latency_ms']} | — |"
    )
    return lines


def _infraction_table(report: dict) -> list[str]:
    return infraction_table(
        [
            ("Baseline", report["baseline"]["summary"]["infraction_totals"]),
            ("Candidate", report["candidate"]["summary"]["infraction_totals"]),
        ],
        empty_message="Neither arm committed a scoreable or tracked infraction.",
    )


def _per_episode_table(report: dict) -> list[str]:
    spec_by_id = {spec["episode_id"]: spec for spec in report["episodes"]}
    lines = [
        "| Episode | Town | Weather | Seed | Baseline | Candidate | Delta | Flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["difference"]["per_episode_driving_score"]:
        spec = spec_by_id[row["episode_id"]]
        flag = (
            f"baseline {row['baseline_status']}, candidate {row['candidate_status']}"
            if "baseline_status" in row
            else ""
        )
        lines.append(
            f"| {row['episode_id']} | {spec['town']} | {spec['weather']} | {spec['seed']} "
            f"| {row['baseline']} | {row['candidate']} | {row['delta']} | {flag} |"
        )
    return lines


def _binding_constraint_verdict(report: dict) -> list[str]:
    """The recorded — never acted-on — fine-tuning trigger. The rule itself
    is stated once, in the rendered "Rule" paragraph below."""
    failed = set(report["difference"]["failed_episodes"])
    clean = [
        row
        for row in report["difference"]["per_episode_driving_score"]
        if row["episode_id"] not in failed
    ]
    lines = [
        "## Is perception the binding constraint?",
        "",
        "Rule, fixed before the run: perception is the **binding constraint** on "
        "driving score when the score it costs (baseline minus candidate) exceeds "
        "the baseline's own shortfall from a perfect 100 — that is, when imperfect "
        "perception loses more score than everything else in the stack combined. "
        "The rule is conservative under a weak baseline: the lower the baseline's "
        "own score, the more perception must cost before it counts as binding. "
        "Episodes flagged as failed measure a mid-run failure, not perception, and "
        "are excluded from this verdict. The verdict is computed from the report "
        "artifact at render time, so revising the rule only means regenerating "
        "this file — never re-running the simulation.",
        "",
    ]
    if failed:
        lines += [f"Excluded failed episode(s): {', '.join(sorted(failed))}.", ""]
    if not clean:
        lines += [
            "Every episode failed mid-run; the verdict cannot be computed from "
            "this suite.",
        ]
        return lines

    baseline_mean = statistics.fmean(row["baseline"] for row in clean)
    candidate_mean = statistics.fmean(row["candidate"] for row in clean)
    cost = baseline_mean - candidate_mean
    shortfall = 100.0 - baseline_mean
    lines += [
        f"Over the {len(clean)} clean episode(s): perception cost "
        f"{cost:.2f} driving-score points; the baseline's own shortfall is "
        f"{shortfall:.2f} points.",
        "",
    ]
    if cost > shortfall:
        lines += [
            "**Perception is the binding constraint on driving score in this "
            "run.** Per issue #10, this is recorded as triggering the deferred "
            "Detector fine-tuning decision — recorded, not acted on. No "
            "fine-tuning has been performed.",
        ]
    else:
        lines += [
            "**Perception is not the binding constraint on driving score in "
            "this run** — the controller and the rest of the stack cost more "
            "than perception did. The deferred Detector fine-tuning decision "
            "is not triggered by this run.",
        ]
    return lines


def render_writeup(report: dict, *, source: str) -> str:
    """Render one ablation report dict as a self-labelling Markdown write-up.

    Args:
        report: A ``run_ablation`` report, exactly as serialised to JSON.
        source: Path of the report artifact, as the write-up should cite it —
            the traceability link from every number back to the run.
    """
    driving_quality = report["scope"] == SCOPE_DRIVING_QUALITY

    lines = [
        f"# Perception ablation — {report['backend']} backend",
        "",
        generated_from(
            report, source=source, generator="pathfinder/ablation_writeup.py"
        ),
        "",
        scope_banner(report),
        "",
        "## What differs between the arms",
        "",
        "Both arms drive identical seeded episodes — same routes, traffic, "
        "weather, controller "
        f"(`{report['controller']}`), and scoring — and differ only in "
        "perception. The perception boundary is: "
        f"**{report['perception_boundary']}**. The Detector arm earns obstacle "
        "ranges from camera pixels; localization and traffic lights stay "
        "privileged in *both* arms because the Detector has no traffic-light "
        "class and localization is a map problem. Claiming more than this "
        "boundary would be false.",
        "",
        *suite_section(report["episodes"], episodes_label="Episodes per arm"),
        "",
        "## Results",
        "",
        *_results_table(report),
        "",
        "A positive difference means the candidate's perception cost driving "
        "performance.",
        "",
        "## Infraction breakdown",
        "",
        *_infraction_table(report),
        "",
        "## Per-episode driving score",
        "",
        *_per_episode_table(report),
    ]

    failed = report["difference"]["failed_episodes"]
    if failed:
        lines += [
            "",
            f"Episode(s) {', '.join(failed)} failed mid-run in at least one arm; "
            "their deltas measure the failure, not perception, and are excluded "
            "from any perception-cost claim.",
        ]

    if report["backend"] == "carla":
        lines += [
            "",
            "The Detector was trained only on real KITTI imagery, so every CARLA "
            "town and weather above is unseen by it — synthetic renders are a "
            "genuine domain shift, and a large gap is the expected outcome. The "
            "measurement is the deliverable either way.",
            "",
            "## Known measurement divergences",
            "",
            "Both arms measure the same range convention (forward-frustum, "
            "camera-origin, ground-plane distance to the obstacle's nearest "
            "visible surface; issue #10), so the gap between them is perception "
            "— with one floor worth naming: below the camera's "
            "`min_measurable_range_m` the obstacle's ground contact leaves the "
            "frame, the Detector's range saturates and over-reads, while the "
            "privileged arm still reads the true range. That divergence *is* a "
            "perception limitation and, alongside detection quality itself, is "
            "part of the measured cost.",
        ]

    if driving_quality:
        lines += ["", *_binding_constraint_verdict(report)]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    return regenerate_writeup_main(
        argv,
        kind="perception_ablation",
        render_writeup=render_writeup,
        description="Regenerate the Markdown write-up for an ablation report.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
