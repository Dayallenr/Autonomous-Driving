"""
Shared machinery for the run-and-report entry points.

``ablation``, ``comparison``, and ``reference_run`` all follow the same
contract: run a seeded suite, checkpoint every finished Episode against a
crash, land one JSON report plus a generated Markdown write-up, and label in
the artifact itself what the numbers are allowed to mean. By the time the
reference run existed, each of those pieces had three near-copies (the Rule
of Three, issue #29); this module is their one home. A fourth report kind —
the Phase 5 distributed-run report is the candidate — should start here
rather than copy a sibling.

Each report module still owns its *sentences* (what its scope note says, what
its write-up claims); this module owns the *shapes* those sentences travel
in, so a fix to the machinery lands everywhere at once.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathfinder.metrics.driving_score import EpisodeScore

__all__ = [
    "NOT_PROJECT_WORK",
    "SCOPE_DRIVING_QUALITY",
    "SCOPE_PIPELINE_ONLY",
    "ReportArtifact",
    "generated_from",
    "infraction_table",
    "print_cli_tail",
    "regenerate_writeup_main",
    "scope",
    "scope_banner",
    "suite_section",
    "unique_in_order",
]

SCOPE_DRIVING_QUALITY = "driving-quality"
SCOPE_PIPELINE_ONLY = "pipeline-only"

#: The sentence that travels with the behaviour agent in every write-up that
#: cites it. (``reference_run.py`` carries a plain-text sibling inside its
#: report JSON, so the disclaimer survives even when only the JSON does.)
NOT_PROJECT_WORK = (
    "CARLA's own behaviour agent — **not project work**. It reads the "
    "simulator's world state directly and appears only as a reference upper "
    "bound produced by the same routes, traffic, and scoring; nothing it "
    "does may be presented as this project's driving."
)


class RenderWriteup(Protocol):
    def __call__(self, report: dict, *, source: str) -> str: ...


def scope(
    backend_name: str,
    *,
    driving_quality_note: str,
    pipeline_limits: str,
    pipeline_label: str,
) -> tuple[str, str]:
    """What a report's numbers are allowed to mean.

    Only CARLA earns the driving-quality label; anything else — including
    backends that do not exist yet — defaults to pipeline-only, because the
    safe failure mode for an unknown backend is underclaiming.

    Args:
        backend_name: The simulator backend the report ran on.
        driving_quality_note: The report kind's own sentence for a CARLA run.
        pipeline_limits: Why a non-CARLA run cannot claim driving quality,
            completing "the {backend} backend, which {pipeline_limits}".
        pipeline_label: The pipeline a non-CARLA run verifies, completing
            "These numbers verify the {pipeline_label} end to end".
    """
    if backend_name == "carla":
        return (SCOPE_DRIVING_QUALITY, driving_quality_note)
    return (
        SCOPE_PIPELINE_ONLY,
        f"Generated on the {backend_name} backend, which {pipeline_limits}. "
        f"These numbers verify the {pipeline_label} end to end; they are not "
        "driving quality and must never be quoted as such. The real "
        "measurement comes from the CARLA backend.",
    )


class ReportArtifact:
    """One run's report JSON + write-up pair, with per-episode crash insurance.

    Construction claims the output path: the parent directory exists
    afterwards and any stale ``.partial.jsonl`` from an earlier run is gone.
    :meth:`checkpoint` is the run's ``on_episode`` callback — every finished
    Episode lands in the partial immediately, so a crash late in an hours-long
    CARLA suite costs one episode, not the sitting. :meth:`finish` writes the
    full report, retires the partial (a lingering partial would mean the run
    did not finish), and renders the sibling write-up — so a CARLA sitting can
    never end with numbers but no document stating what they may mean.

    Every write passes ``encoding="utf-8"`` explicitly. This is not optional:
    ``write_text`` defaults to the locale encoding, which is cp1252 on
    Windows, and the rendered write-ups contain U+2212 MINUS SIGN — without it
    a finished 20-episode CARLA run died at the very last line, leaving an
    empty ``.md`` beside a valid ``.json``.
    """

    def __init__(self, output: Path, *, label_key: str) -> None:
        self.output = output
        self.label_key = label_key
        self.partial = output.with_suffix(".partial.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        self.partial.unlink(missing_ok=True)

    def checkpoint(self, label: str, score: EpisodeScore) -> None:
        """Append one scored Episode, attributed under ``label_key``."""
        with self.partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({self.label_key: label, **score.to_dict()}) + "\n")

    def finish(self, report: dict, render_writeup: RenderWriteup) -> Path:
        """Land the report, retire the partial, render the write-up.

        The partial is removed the moment the full report exists — before the
        write-up renders — so a render failure cannot leave a partial lying
        around implying the suite itself did not finish.

        Returns:
            The write-up's path (the report's sibling ``.md``).
        """
        self.output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        self.partial.unlink(missing_ok=True)
        writeup = self.output.with_suffix(".md")
        writeup.write_text(
            render_writeup(report, source=str(self.output)), encoding="utf-8"
        )
        return writeup


def print_cli_tail(report: dict, output: Path, writeup: Path) -> None:
    """The lines every report CLI ends with: the pipeline-only warning when it
    applies, and where both artifacts landed."""
    if report["scope"] == SCOPE_PIPELINE_ONLY:
        print("NOTE: pipeline-only run — these numbers are not driving quality.")
    print(f"report written to {output}")
    print(f"write-up written to {writeup}")


def unique_in_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def generated_from(report: dict, *, source: str, generator: str) -> str:
    """The traceability paragraph every write-up opens with: which artifact
    the numbers come from, and which module renders it. The link target is the
    report's bare file name because the write-up lands beside it."""
    return (
        f"Generated {report['generated_at']} from [`{source}`]"
        f"({Path(source).name}), which records every number below along with "
        "the full episode specifications needed to re-run it. This file is "
        f"rendered from that artifact by `{generator}`; edit "
        "the generator, not this file."
    )


def scope_banner(report: dict) -> str:
    """The blockquote that pins the report's own scope label to the top of
    its write-up."""
    return f"> **Scope: {report['scope']}.** {report['scope_note']}"


def suite_section(specs: list[dict], *, episodes_label: str) -> list[str]:
    """The write-up's Suite section, from the report's recorded episode specs.

    Args:
        specs: The report's ``episodes`` list (serialised ``EpisodeSpec``s).
        episodes_label: The first bullet's label — "Episodes per arm",
            "Episodes per column", or plain "Episodes".
    """
    # An empty suite still gets a section: a report kind whose episodes come
    # from a queue can legitimately finish with none, and its write-up must
    # land beside the JSON rather than die summarising nothing.
    if not specs:
        return ["## Suite", "", f"- {episodes_label}: 0"]
    towns = unique_in_order([spec["town"] for spec in specs])
    weathers = unique_in_order([spec["weather"] for spec in specs])
    seeds = [spec["seed"] for spec in specs]
    return [
        "## Suite",
        "",
        f"- {episodes_label}: {len(specs)}",
        f"- Towns: {', '.join(towns)}",
        f"- Weathers: {', '.join(weathers)}",
        f"- Seeds: {min(seeds)}–{max(seeds)}",
    ]


def infraction_table(
    columns: list[tuple[str, dict[str, int] | None]], *, empty_message: str
) -> list[str]:
    """A Markdown table of infraction counts, one column per arm.

    Args:
        columns: ``(heading, infraction_totals)`` per column; ``None`` totals
            mark a column that did not run and render as "not run".
        empty_message: The single line rendered when no column committed any
            scoreable or tracked infraction.
    """
    names = sorted(
        {name for _, totals in columns if totals is not None for name in totals}
    )
    if not names:
        return [empty_message]
    lines = [
        "| Infraction | " + " | ".join(heading for heading, _ in columns) + " |",
        "|---|" + "---|" * len(columns),
    ]
    for name in names:
        cells = [
            "not run" if totals is None else str(totals.get(name, 0))
            for _, totals in columns
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def regenerate_writeup_main(
    argv: list[str] | None,
    *,
    kind: str,
    render_writeup: RenderWriteup,
    description: str,
) -> int:
    """The write-up regenerator CLI shared by every ``*_writeup`` module:
    re-render an existing report's sibling ``.md`` without re-running
    anything. Refuses a report of any other kind — rendering a report through
    the wrong generator would produce a document making claims the artifact
    does not back."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("report", type=Path, help=f"path to a {kind} JSON report")
    args = parser.parse_args(argv)

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("kind") != kind:
        raise SystemExit(f"{args.report} is not a {kind} report")
    output = args.report.with_suffix(".md")
    # utf-8 is explicit for the same reason as in ReportArtifact.finish: the
    # rendered write-up is not ASCII, and the locale default is cp1252 on
    # Windows.
    output.write_text(render_writeup(report, source=str(args.report)), encoding="utf-8")
    print(f"write-up written to {output}")
    return 0
