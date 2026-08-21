"""
Render a DAgger run report artifact into its Markdown write-up.

Generated from the report JSON rather than written by hand for the same reason
as the ablation's write-up: issue #20 puts hard requirements on it — a
smoke-is-not-a-result banner whenever the configuration is below the
documented real-run bar, the disagreement-direction caveat either way, and
traceability to the producing run — and a generated document keeps those
properties under test instead of under memory.

    python -m pathfinder.dagger_writeup results/dagger/carla/report.json

regenerates the sibling ``.md`` from an existing report. The DAgger CLI calls
:func:`render_writeup` itself after every run, so normally the two artifacts
land together.
"""
from __future__ import annotations

from pathlib import Path

from pathfinder.policies import CarlaBehaviorAgentPolicy
from pathfinder.reporting import regenerate_writeup_main, scope_banner

__all__ = ["render_writeup"]

#: The sentence that travels with the behaviour agent wherever it appears.
_NOT_PROJECT_WORK = (
    "CARLA's own behaviour agent — **not project work**. It appears here as "
    "the expert labeller only; nothing it does may be presented as this "
    "project's driving. The trained student is the project work."
)


def _setup_lines(report: dict) -> list[str]:
    config = report["config"]
    student = report["student"]
    expert_line = f"- Expert (labels every visited state): `{report['expert']}`"
    if report["expert"] == CarlaBehaviorAgentPolicy.NAME:
        expert_line += f". {_NOT_PROJECT_WORK}"
    return [
        expert_line,
        f"- Student: `{student['model']}` ({student['output_mode']} output, "
        f"ImageNet-pretrained: {'yes' if student['imagenet_pretrained'] else 'no'})",
        f"- Backend `{report['backend']}`, device `{config['device']}`, seed {config['seed']}",
        f"- β schedule: 1.0 at iteration 0, then {config['beta_decay']}^iteration — "
        "β is the probability the expert's action drives",
        f"- {config['iterations']} iterations × {config['episodes_per_iteration']} "
        f"episode(s) of {config['route_length_m']} m (≤ {config['max_steps']} steps); "
        f"{config['epochs_per_iteration']} epoch(s) per iteration, batch "
        f"{config['batch_size']}, learning rate {config['learning_rate']}, dataset "
        f"cap {config['max_dataset_size']}",
        f"- Towns: {', '.join(config['towns'])}; weathers: {', '.join(config['weathers'])} "
        "(both ignored by the kinematic backend)",
    ]


def _bar_table(report: dict) -> list[str]:
    lines = [
        "| Criterion | Required | Actual | Met |",
        "|---|---|---|---|",
    ]
    for name, item in report["real_run_bar"]["criteria"].items():
        required = item["required"]
        if isinstance(required, int) and not isinstance(required, bool):
            required = f"≥ {required}"
        lines.append(
            f"| {name} | {required} | {item['actual']} "
            f"| {'yes' if item['met'] else '**no**'} |"
        )
    return lines


def _iteration_table(report: dict) -> list[str]:
    lines = [
        "| Iteration | β | Dataset size | Train loss | Disagreement | "
        "Rollout score | Rollout completion | Seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["iterations"]:
        disagreement = "—" if row["disagreement"] is None else row["disagreement"]
        lines.append(
            f"| {row['iteration']} | {row['beta']} | {row['dataset_size']} "
            f"| {row['train_loss']} | {disagreement} | {row['driving_score']} "
            f"| {row['route_completion']} | {row['seconds']} |"
        )
    return lines


def render_writeup(report: dict, *, source: str) -> str:
    """Render one DAgger run report dict as a self-labelling Markdown write-up.

    Args:
        report: A ``build_run_report`` report, exactly as serialised to JSON.
        source: Path of the report artifact, as the write-up should cite it.
    """
    lines = [
        f"# DAgger run — {report['backend']} backend",
        "",
        f"Generated {report['generated_at']} from [`{source}`]"
        f"({Path(source).name}), which records every number below along with "
        "the full configuration needed to re-run it. This file is rendered "
        "from that artifact by `pathfinder/dagger_writeup.py`; edit the "
        "generator, not this file.",
        "",
        scope_banner(report),
        "",
        "## Setup",
        "",
        *_setup_lines(report),
        "",
        "## Real-run bar",
        "",
        "A run's figures may be read as training results only when every "
        "criterion below is met; otherwise the whole document is a loop check.",
        "",
        *_bar_table(report),
        "",
        "## Per-iteration results",
        "",
        *_iteration_table(report),
        "",
        f"*{report['iteration_metrics_note']}*",
        "",
        "## Reading the disagreement numbers",
        "",
        "Disagreement is the mean L1 gap between the expert's and the "
        "student's controls, measured at every state the iteration's rollouts "
        "visit. Those rollouts run under the β-mixed policy, so early "
        "iterations measure it mostly on expert-visited states; as β decays "
        "the states are increasingly the student's own — the distribution "
        "DAgger is designed to reduce error on. Iteration 0 rolls out the "
        "expert alone, so it has no measurement.",
        "",
        "Two effects pull the number in opposite directions as a run "
        "proceeds: as β decays the student drives more and reaches worse "
        "states, where the expert's corrective actions are more extreme — "
        "pushing disagreement **up**; as the network fits the growing "
        "aggregate dataset it predicts those corrections better — pushing it "
        "**down**. The second effect dominates only once the model has the "
        "data and epochs to actually fit. So disagreement rising across "
        "early iterations does not by itself mean the loop is broken, and "
        "disagreement falling does not by itself certify a strong student — "
        "driving quality is measured by the scored comparison suite, never "
        "by this table.",
        "",
        (
            "First-to-last change in measured disagreement on this run: not "
            "measurable — fewer than two iterations produced a measurement, "
            "or the first measurement was zero."
            if report["error_reduction_pct"] is None
            else f"First-to-last change in measured disagreement on this run: "
            f"{report['error_reduction_pct']}% (positive means it fell)."
        ),
        "",
        "## Artifacts",
        "",
        f"- Final weights: `{report['weights']}`, beside this file; one "
        "checkpoint per iteration, each holding model, optimizer, and its "
        "report row.",
        "- Per-iteration sample archives are deleted when a run completes; "
        "their absence marks a finished run.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    return regenerate_writeup_main(
        argv,
        kind="dagger_run",
        render_writeup=render_writeup,
        description="Regenerate the Markdown write-up for a DAgger run report.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
