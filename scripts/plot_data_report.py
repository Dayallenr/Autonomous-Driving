#!/usr/bin/env python
"""
Render the Phase 1 data-engineering figures from ``data/manifest.json``.

    python scripts/prepare_kitti.py --report-baseline   # writes the manifest
    python scripts/plot_data_report.py                  # writes the figures

Figures land in ``results/data/`` in light and dark variants, so a README can
serve the right one via ``<picture>``. Everything is derived from the manifest —
no number is typed into this file, so a figure cannot drift from the split that
produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Validated categorical slots 1 and 2 (see the data-viz palette reference).
# Checked with the palette validator: all-pairs CVD delta-E 24.7, normal-vision
# 33.6, both well clear of the floors, in light and dark.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ("#2a78d6", "#eb6834"),
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ("#3987e5", "#d95926"),
    },
}


def style(theme: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "text.color": theme["primary"],
        "axes.labelcolor": theme["secondary"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "axes.edgecolor": theme["axis"],
        "grid.color": theme["grid"],
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 160,
    })


def finish(axes, theme: dict, title: str, subtitle: str = "") -> None:
    """Titles as left-aligned text, so they read as prose rather than captions."""
    axes.set_title("")
    axes.text(
        0.0, 1.14, title, transform=axes.transAxes,
        fontsize=13, fontweight="600", color=theme["primary"], ha="left",
    )
    if subtitle:
        axes.text(
            0.0, 1.05, subtitle, transform=axes.transAxes,
            fontsize=9.5, color=theme["secondary"], ha="left",
        )


def plot_leakage(manifest: dict, theme: dict, out: Path) -> None:
    """The measurement that justifies the whole split: near-duplicate val frames.

    A grouped bar rather than two panels — the comparison *is* the point, and
    side-by-side bars put the two conditions on one shared scale where the
    baseline's height and the rebuilt split's flat zero read in a single glance.
    """
    baseline = manifest["baseline_random_split"]["temporal_adjacency"]
    ours = manifest["leakage"]["temporal_adjacency"]
    gaps = sorted(baseline, key=int)

    figure, axes = plt.subplots(figsize=(7.6, 4.0))
    positions = range(len(gaps))
    width = 0.38

    baseline_values = [baseline[g]["fraction"] * 100 for g in gaps]
    our_values = [ours[g]["fraction"] * 100 for g in gaps]

    axes.bar(
        [p - width / 2 for p in positions], baseline_values, width,
        label="Random split (Ultralytics kitti.yaml)",
        color=theme["series"][1], edgecolor=theme["surface"], linewidth=2,
    )
    axes.bar(
        [p + width / 2 for p in positions], our_values, width,
        label="Sequence-disjoint split (this repo)",
        color=theme["series"][0], edgecolor=theme["surface"], linewidth=2,
    )

    for position, value in zip(positions, baseline_values, strict=True):
        axes.text(
            position - width / 2, value + 1.8, f"{value:.0f}%",
            ha="center", fontsize=9, color=theme["secondary"], fontweight="600",
        )

    # A zero-height bar draws nothing, which reads as a series that failed to
    # render rather than one that measured zero. Label the zeros explicitly so
    # the absence is stated as a result.
    for position, value in zip(positions, our_values, strict=True):
        axes.text(
            position + width / 2, 1.8, f"{value:.0f}%",
            ha="center", fontsize=9, color=theme["series"][0], fontweight="600",
        )

    axes.set_xticks(list(positions))
    axes.set_xticklabels([f"±{g}" for g in gaps])
    axes.set_xlabel("Maximum gap to a training frame, in raw 10 Hz video frames")
    axes.set_ylabel("Validation frames affected")
    axes.set_ylim(0, 105)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    axes.set_axisbelow(True)
    axes.grid(axis="x", visible=False)

    legend = axes.legend(
        frameon=False, fontsize=9, loc="upper left",
        bbox_to_anchor=(0.0, -0.20), ncol=1, handlelength=1.2, handleheight=0.9,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    finish(
        axes, theme,
        "A random split over KITTI leaks the validation set",
        "±1 frame is 0.1 s apart — effectively the same photograph. "
        "Splitting by source drive removes it entirely.",
    )
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution(manifest: dict, theme: dict, out: Path) -> None:
    """Instance counts per class, log-scaled.

    Log scale because the range is 129:1 — on a linear axis every class except
    ``car`` collapses into the axis and the imbalance, which is the entire
    reason the rebalancing step exists, becomes invisible.
    """
    distribution = manifest["class_distribution"]
    names = list(distribution["train"])
    train = [distribution["train"][n]["instances"] for n in names]
    val = [distribution["val"][n]["instances"] for n in names]

    order = sorted(range(len(names)), key=lambda i: train[i] + val[i])
    names = [names[i] for i in order]
    train = [train[i] for i in order]
    val = [val[i] for i in order]

    figure, axes = plt.subplots(figsize=(7.6, 4.4))
    positions = range(len(names))
    height = 0.38

    axes.barh(
        [p + height / 2 for p in positions], train, height, label="train",
        color=theme["series"][0], edgecolor=theme["surface"], linewidth=2,
    )
    axes.barh(
        [p - height / 2 for p in positions], val, height, label="val",
        color=theme["series"][1], edgecolor=theme["surface"], linewidth=2,
    )

    for position, (t, v) in enumerate(zip(train, val, strict=True)):
        axes.text(t * 1.12, position + height / 2, f"{t:,}", va="center",
                  fontsize=8.5, color=theme["secondary"])
        axes.text(v * 1.12, position - height / 2, f"{v:,}", va="center",
                  fontsize=8.5, color=theme["muted"])

    axes.set_xscale("log")
    axes.set_yticks(list(positions))
    axes.set_yticklabels(names)
    axes.set_xlabel("Instances (log scale)")
    axes.set_xlim(10, max(train) * 4)
    axes.set_axisbelow(True)
    axes.grid(axis="y", visible=False)

    legend = axes.legend(frameon=False, fontsize=9, loc="lower right", handlelength=1.2)
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    # Dataset-level imbalance, over both splits — the property of KITTI being
    # described. Computing it on train alone would report an artefact of the
    # split instead.
    totals = [t + v for t, v in zip(train, val, strict=True)]
    ratio = max(totals) / max(min(totals), 1)
    finish(
        axes, theme,
        "KITTI is severely class-imbalanced",
        f"{ratio:,.0f}:1 between the most and least common class — "
        "which is why rare-class images are oversampled during training.",
    )
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def plot_val_share(manifest: dict, theme: dict, out: Path) -> None:
    """Per-class validation share against the 20% target.

    A dot plot, not bars: the quantity that matters is each class's *distance
    from a target*, and dots against a reference line show deviation directly
    while bars would encode the share's magnitude — the wrong variable.
    """
    distribution = manifest["class_distribution"]
    target = manifest["split"]["val_fraction"]

    names, shares = [], []
    for name in distribution["train"]:
        train = distribution["train"][name]["instances"]
        val = distribution["val"][name]["instances"]
        if train + val:
            names.append(name)
            shares.append(val / (train + val) * 100)

    order = sorted(range(len(names)), key=lambda i: shares[i])
    names = [names[i] for i in order]
    shares = [shares[i] for i in order]

    figure, axes = plt.subplots(figsize=(7.6, 4.0))
    positions = list(range(len(names)))

    axes.axvline(target * 100, color=theme["axis"], linewidth=1.4, zorder=1)
    axes.text(
        target * 100, -0.85, f"target {target * 100:.0f}%",
        fontsize=8.5, color=theme["muted"], va="center", ha="center",
    )

    for position, share in zip(positions, shares, strict=True):
        axes.plot(
            [target * 100, share], [position, position],
            color=theme["grid"], linewidth=1.6, zorder=2, solid_capstyle="round",
        )
    axes.scatter(
        shares, positions, s=90, color=theme["series"][0],
        edgecolor=theme["surface"], linewidth=1.8, zorder=3,
    )
    # Label on the far side of the dot from the target line, so a value close to
    # target never prints its text across the reference.
    for position, share in zip(positions, shares, strict=True):
        below = share < target * 100
        axes.text(
            share + (-0.9 if below else 0.9), position, f"{share:.1f}%",
            va="center", ha="right" if below else "left",
            fontsize=8.5, color=theme["secondary"],
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(names)
    axes.set_xlabel("Share of each class's instances landing in validation")
    # Zoomed to the occupied range. On a dot plot the encoded quantity is
    # distance from the reference line, not bar length from zero, so a window
    # around the target is the honest framing — anchoring at 0 would spend two
    # thirds of the axis on empty space and flatten every deviation to nothing.
    axes.set_xlim(15, max(shares) + 5)
    axes.set_xticks([15, 20, 25, 30])
    axes.set_xticklabels(["15%", "20%", "25%", "30%"])
    axes.set_ylim(-1.4, len(names) - 0.4)
    axes.set_axisbelow(True)
    axes.grid(axis="y", visible=False)

    finish(
        axes, theme,
        "Every class keeps usable validation support",
        "Drives are indivisible, so shares cannot land exactly on target. "
        "person_sitting occupies 3 of 141 drives — 25.2% is its closest reachable value.",
    )
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Phase 1 data figures")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--out", type=Path, default=Path("results/data"))
    arguments = parser.parse_args()

    if not arguments.manifest.exists():
        print(
            f"no manifest at {arguments.manifest}\n"
            "run: python scripts/prepare_kitti.py --report-baseline",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    if "baseline_random_split" not in manifest:
        print(
            "manifest has no baseline measurement; "
            "re-run prepare_kitti.py with --report-baseline",
            file=sys.stderr,
        )
        return 1

    arguments.out.mkdir(parents=True, exist_ok=True)
    written = []
    for mode, theme in THEMES.items():
        style(theme)
        suffix = "" if mode == "light" else "-dark"
        for name, render in (
            ("split_leakage", plot_leakage),
            ("class_distribution", plot_class_distribution),
            ("val_share", plot_val_share),
        ):
            path = arguments.out / f"{name}{suffix}.png"
            render(manifest, theme, path)
            written.append(path)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
