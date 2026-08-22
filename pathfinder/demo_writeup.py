"""
Render a ``demo_capture`` report (``pathfinder/demo.py``, issue #27) into its
write-up: the README paste block.

For a CARLA capture the write-up carries a fenced Markdown block to paste into
the README verbatim — the embed plus a caption naming the Policy, the seed,
and the Episode's own score, every number cited in the ``docs/CLAIMS.md``
convention against the report, and the media files cited as prose claims. The
fence keeps the citations inert in this file (their paths are README-relative,
not write-up-relative) while the claim checker enforces them the moment they
land in the README. A pipeline-only capture gets no paste block at all: a
kinematic clip in the README would present a physics-free render as the demo,
so the write-up says exactly that instead.

Standalone regeneration, like every sibling write-up module:

    python -m pathfinder.demo_writeup results/demo/carla_demo.json

so a wording fix here never requires re-driving a ten-minute CARLA Episode.
"""
from __future__ import annotations

from pathlib import PurePosixPath

from pathfinder import reporting

__all__ = ["extract_paste_block", "render_writeup"]

_PASTE_FENCE = "```markdown\n"


def _cite(value: object, artifact: str, field_path: str) -> str:
    """One machine-checked citation in the ``docs/CLAIMS.md`` convention.

    The quote is ``str(value)`` of the artifact's own (already rounded) JSON
    value, so the checker's decimal-count comparison matches by construction —
    quoting a re-rounded number here would be the drift the checker exists to
    catch."""
    return f'[{value}]({artifact} "claim:{field_path}")'


def extract_paste_block(writeup: str) -> str | None:
    """The fenced block the user pastes into the README, or None when the
    write-up carries none (every pipeline-only capture). Shared by the capture
    CLI, which prints the block, and the tests that hold it to the claim
    checker — one definition of where the block begins and ends."""
    if _PASTE_FENCE not in writeup:
        return None
    return writeup.split(_PASTE_FENCE)[1].split("```")[0]


def render_writeup(report: dict, *, source: str) -> str:
    """Render a ``demo_capture`` report's write-up (see the module docstring
    for what a CARLA versus a pipeline-only capture gets)."""
    # The capture runs in PowerShell, where the report path arrives with
    # backslashes. The paste block's paths must be forward-slashed regardless:
    # GitHub cannot resolve backslashed links and the claim checker on Linux
    # CI could not find the artifacts.
    source = str(source).replace("\\", "/")
    directory = PurePosixPath(source).parent
    episode, score, media, policy = (
        report["episode"], report["score"], report["media"], report["policy"],
    )
    lines = [
        "# Demo capture",
        "",
        reporting.generated_from(report, source=source, generator="pathfinder.demo"),
        "",
        reporting.scope_banner(report),
        "",
        f"- Episode: `{episode['episode_id']}` — {episode['town']}, "
        f"{episode['weather']}, seed {episode['seed']}",
        f"- Policy: `{policy['model_version']}` with {policy['perception']} perception",
        f"- Driving score {score['driving_score']}, route completion "
        f"{score['route_completion']} of {episode['route_length_m']} m, "
        f"{score['frames']} frames ({score['termination_reason']})",
        f"- Media: `{media['video']}` ({media['frames']} frames at "
        f"{media['fps']:g} fps), `{media['gif']}` (first {media['gif_covers_seconds']} s)",
        "",
    ]

    if report["scope"] != reporting.SCOPE_DRIVING_QUALITY:
        lines += [
            "This capture is **never the README demo**: only a CARLA Episode may "
            "be embedded there, and this one verified the capture pipeline on the "
            f"`{report['backend']}` backend. Record the real clip with "
            "`python -m pathfinder.demo --backend carla` (docs/SETUP_WINDOWS.md §10).",
            "",
        ]
        return "\n".join(lines)

    gif = str(directory / media["gif"])
    video = str(directory / media["video"])
    lines += [
        "Paste the block below into the README's demo slot (the issue #27 "
        "placeholder), replacing it. Paths are README-relative, so the report "
        "and media must live where this capture wrote them, committed to the "
        "repo. The claim checker then verifies every number and the media "
        "files' existence in CI.",
        "",
        _PASTE_FENCE.rstrip("\n"),
        f'<img src="{gif}" alt="{policy["model_version"]} driving a seeded CARLA '
        f'Episode in {episode["town"]}">',
        "",
        f"*A seeded CARLA Episode — {episode['town']}, {episode['weather']}, seed "
        f"{_cite(episode['seed'], source, 'episode.seed')} — driven by this "
        f"project's `{policy['model_version']}` with {policy['perception']} "
        "perception (the ablation's baseline arm): driving score "
        f"{_cite(score['driving_score'], source, 'score.driving_score')}, "
        "completing "
        f"{_cite(score['route_completion'], source, 'score.route_completion')} "
        "of its "
        f"{_cite(episode['route_length_m'], source, 'episode.route_length_m')} m "
        "route. The GIF shows the first "
        f"{_cite(media['gif_covers_seconds'], source, 'media.gif_covers_seconds')} s "
        f"in real time; [the full clip]({video} \"claim:prose\") and "
        f"[the capture report]({source} \"claim:prose\") sit beside "
        f"[it]({gif} \"claim:prose\"). Recorded by `python -m pathfinder.demo "
        "--backend carla`.*",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    return reporting.regenerate_writeup_main(
        argv,
        kind="demo_capture",
        render_writeup=render_writeup,
        description="Regenerate a demo_capture report's write-up (the README paste block).",
    )


if __name__ == "__main__":
    raise SystemExit(main())
