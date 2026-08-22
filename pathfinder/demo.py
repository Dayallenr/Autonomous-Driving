"""
The demo capture (issue #27): record one seeded Episode as a clip the README
can embed, with a sidecar report the caption cites through the claim checker.

    python -m pathfinder.demo --backend carla

One command replays a single Episode from the ablation's exact suite (by
default ``ep-0000``: Town01, ClearNoon, seed 1000) under the same privileged
PurePursuit arm the ablation's baseline ran, records it to an MP4 plus a
README-embeddable GIF, and lands a ``demo_capture`` report JSON beside them.
The generated write-up *is* the README paste block: a caption naming the
Policy, the seed, and the Episode's own score, every number cited in the
``docs/CLAIMS.md`` convention and the media files cited as prose claims — so
once pasted, the claim checker enforces that the clip exists and the caption
matches the run that produced it.

Scope follows the shared rule (:func:`pathfinder.reporting.scope`): only a
CARLA capture may be embedded as the demo. A kinematic capture exercises this
whole pipeline on a machine without CARLA, and its report and write-up label
themselves pipeline-only — never the README demo.
"""
from __future__ import annotations

import argparse
import math
import queue
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pathfinder import reporting
from pathfinder.perception.privileged import PrivilegedPerception
from pathfinder.policies import ModularPolicy
from pathfinder.runner import ControlOutput, Policy, PurePursuitPolicy, run_episode
from pathfinder.sim.base import EpisodeSpec, FrameState, SimulatorBackend

__all__ = ["ChaseCamera", "DemoRecorder", "capture_demo", "render_snippet", "suite_episode"]

#: What a frame source yields: the frame to record for the current state, or
#: None when no frame is available this tick (a sensor queue can lag a step).
FrameSource = Callable[[FrameState], np.ndarray | None]


class DemoRecorder:
    """Encode an Episode's frames into an MP4 and a subsampled GIF.

    The MP4 gets every frame, written incrementally so memory stays bounded on
    a 1,500-frame Episode instead of holding the whole clip. The GIF is the
    README-embeddable excerpt: every ``gif_stride``-th frame until
    ``gif_max_frames``, resized to ``gif_width``, with the per-frame duration
    set so it plays back in real time.

    Args:
        stem: Output path without suffix; ``<stem>.mp4`` and ``<stem>.gif``.
        fps: Playback rate — the Episode's tick rate (``1 / delta_seconds``),
            so both files play in real time.
        gif_stride: Keep every Nth frame for the GIF.
        gif_max_frames: Cap on GIF frames; a full Episode as a GIF would be
            tens of megabytes in the README, so the GIF is an excerpt and the
            MP4 is the full record.
        gif_width: GIF width in pixels; height follows the aspect ratio.
        scale: Integer nearest-neighbour upscale applied to every frame.
            Exists for the kinematic backend's 200x88 policy view, which is
            unwatchable at native size; the CARLA chase camera uses 1.
    """

    def __init__(
        self,
        stem: Path,
        *,
        fps: float,
        gif_stride: int,
        gif_max_frames: int,
        gif_width: int,
        scale: int = 1,
    ) -> None:
        if fps <= 0 or gif_stride <= 0 or gif_max_frames <= 0 or scale <= 0:
            raise ValueError("fps, gif_stride, gif_max_frames, and scale must be positive")
        self._stem = stem
        self._fps = fps
        self._gif_stride = gif_stride
        self._gif_max_frames = gif_max_frames
        self._gif_width = gif_width
        self._scale = scale
        self._writer = None
        self._frames = 0
        self._gif_frames: list = []
        self._size: tuple[int, int] | None = None

    def add(self, frame: np.ndarray) -> None:
        """Append one RGB ``(H, W, 3) uint8`` frame."""
        import cv2

        if self._scale > 1:
            frame = frame.repeat(self._scale, axis=0).repeat(self._scale, axis=1)
        height, width = frame.shape[:2]
        if self._writer is None:
            self._size = (width, height)
            self._stem.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self._stem.with_suffix(".mp4")),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self._fps,
                self._size,
            )
        elif (width, height) != self._size:
            raise ValueError(
                f"frame size changed mid-capture: {self._size} -> {(width, height)}"
            )
        # The recorder's contract is RGB in; OpenCV writes BGR.
        self._writer.write(frame[:, :, ::-1])

        if self._frames % self._gif_stride == 0 and len(self._gif_frames) < self._gif_max_frames:
            from PIL import Image

            image = Image.fromarray(frame)
            gif_height = max(1, round(image.height * self._gif_width / image.width))
            self._gif_frames.append(image.resize((self._gif_width, gif_height)))
        self._frames += 1

    def close(self) -> dict:
        """Finish both files and describe them.

        Returns:
            Media metadata for the report: file names (relative to the report,
            which lands beside them), frame counts, dimensions, and how many
            seconds of the Episode the GIF covers. ``frames`` is 0 and no
            files exist when no frame ever arrived.
        """
        if self._writer is not None:
            self._writer.release()
        if not self._frames:
            return {"frames": 0}

        first, *rest = self._gif_frames
        # Per-frame duration in ms: stride ticks of simulated time, so the
        # subsampled GIF still plays back in real time.
        first.save(
            self._stem.with_suffix(".gif"),
            save_all=True,
            append_images=rest,
            duration=round(1000.0 * self._gif_stride / self._fps),
            loop=0,
        )
        width, height = self._size
        return {
            "video": self._stem.with_suffix(".mp4").name,
            "gif": self._stem.with_suffix(".gif").name,
            "frames": self._frames,
            "fps": self._fps,
            "width": width,
            "height": height,
            "gif_frames": len(self._gif_frames),
            "gif_covers_seconds": round(
                min(self._frames, (len(self._gif_frames) - 1) * self._gif_stride + 1)
                / self._fps,
                1,
            ),
        }


class _RecordingPolicy:
    """Record each frame's view, then delegate the driving decision.

    Wrapping the Policy is what keeps :func:`pathfinder.runner.run_episode`
    untouched: ``plan`` already receives every :class:`FrameState`, so the
    recorder sees exactly the frames the Policy planned on, once per tick.
    """

    def __init__(self, inner: Policy, recorder: DemoRecorder, source: FrameSource) -> None:
        self._inner = inner
        self._recorder = recorder
        self._source = source

    def plan(self, state: FrameState) -> ControlOutput:
        frame = self._source(state)
        if frame is not None:
            self._recorder.add(np.asarray(frame))
        return self._inner.plan(state)


def _scope(backend_name: str) -> tuple[str, str]:
    return reporting.scope(
        backend_name,
        driving_quality_note=(
            "Generated on the CARLA backend: the clip shows this project's "
            "Policy driving, and its score is driving quality."
        ),
        pipeline_limits="neither simulates real physics nor renders real scenes",
        pipeline_label="demo capture pipeline",
    )


def capture_demo(
    simulator: SimulatorBackend,
    spec: EpisodeSpec,
    *,
    recorder: DemoRecorder,
    controller_factory: Callable[[], Policy] = PurePursuitPolicy,
    frame_source: FrameSource | None = None,
) -> dict:
    """Drive one Episode, recording what it looked like, and report it.

    The arm is the ablation baseline's exact construction — privileged
    perception feeding the controller through :class:`ModularPolicy` — so the
    captured Episode is the same configuration the ablation's privileged
    numbers came from, not a lookalike.

    Args:
        simulator: Backend to drive in.
        spec: The Episode; drawn from the ablation suite by the CLI so the
            clip replays a benchmarked Episode.
        recorder: Where frames land. Closed by this function — the report
            references the finished files.
        controller_factory: Builds the inner controller, fresh per capture.
        frame_source: What to record each tick; defaults to the state's own
            rendered view (``state.image``). The CARLA chase camera passes a
            source that reads its own sensor instead.

    Returns:
        The ``demo_capture`` report as a JSON-serialisable dict.

    Raises:
        RuntimeError: If no frame was ever recorded — a demo without a clip —
            or if the Episode's frames carry mixed perception provenance.
    """
    source = frame_source if frame_source is not None else lambda state: state.image
    controller = controller_factory()
    model_version = getattr(controller, "NAME", type(controller).__name__)
    policy = _RecordingPolicy(
        ModularPolicy(PrivilegedPerception(), controller), recorder, source
    )

    observed: set[str] = set()

    def sink(row: dict) -> None:
        observed.add(row["perception"])

    score = run_episode(
        simulator, spec, policy, telemetry_sink=sink, model_version=model_version
    )
    media = recorder.close()

    if not media["frames"]:
        raise RuntimeError(
            "no frames were recorded: the backend produced no images to capture "
            "(kinematic needs render=True; carla needs its camera or a chase camera)"
        )
    if len(observed) > 1:
        # Same rule as the ablation: a caption must never attribute the clip
        # to a perception that only ran for part of it.
        raise RuntimeError(f"mixed perception provenance on one Episode: {sorted(observed)}")

    scope, scope_note = _scope(simulator.name)
    return {
        "kind": "demo_capture",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": simulator.name,
        "scope": scope,
        "scope_note": scope_note,
        "policy": {
            "model_version": model_version,
            "perception": observed.pop() if observed else "unobserved",
        },
        "episode": spec.to_dict(),
        "score": score.to_dict(),
        "media": media,
    }


def suite_episode(
    index: int,
    *,
    route_length_m: float = 400.0,
    base_seed: int = 1000,
    max_steps: int = 1500,
) -> EpisodeSpec:
    """The benchmark suite's Episode at ``index`` — the same builder with the
    same defaults the ablation CLI uses, so the demo replays a benchmarked
    Episode rather than a lookalike. ``tests/test_demo.py`` pins this against
    the specs the real CARLA ablation recorded in its artifact.
    """
    if index < 0:
        raise ValueError(f"episode index must be non-negative, got {index}")
    # Imported here for the same reason ablation.py defers it: orchestration
    # pulls in the cloud stack, which building one spec has no business needing.
    from pathfinder.orchestration import build_episode_specs

    return build_episode_specs(
        count=index + 1,
        route_length_m=route_length_m,
        base_seed=base_seed,
        max_steps=max_steps,
    )[index]


class ChaseCamera:
    """Third-person camera for the CARLA capture.

    The Policy's own 200x88 forward camera is what the driving stack sees, but
    it is unwatchable as a demo; this spawns a second, higher-resolution
    camera behind the ego and records that instead. Spawning is lazy — on the
    first frame request — because :func:`pathfinder.runner.run_episode` owns
    ``reset()`` and the ego actor only exists afterwards.

    The BGRA-to-RGB conversion and drain-to-newest mirror the backend's own
    ``_drain_image``; in synchronous mode the sensor delivers one image per
    tick, so recorded frames stay aligned with driven frames (at most one tick
    behind, and ``None`` before the first delivery).
    """

    def __init__(self, simulator, *, width: int = 960, height: int = 540) -> None:
        self._simulator = simulator
        self._width = width
        self._height = height
        self._camera = None
        self._queue: queue.Queue = queue.Queue()

    def source(self, state: FrameState) -> np.ndarray | None:
        """A :data:`FrameSource`: the newest chase-camera frame, as RGB."""
        if self._camera is None:
            self._spawn()
        image = None
        while True:
            try:
                image = self._queue.get_nowait()
            except queue.Empty:
                break
        if image is None:
            return None
        buffer = np.frombuffer(image.raw_data, dtype=np.uint8)
        return buffer.reshape(image.height, image.width, 4)[:, :, :3][:, :, ::-1]

    def _spawn(self) -> None:
        import carla

        vehicle = self._simulator.vehicle
        if vehicle is None:
            raise RuntimeError("no ego vehicle to attach the chase camera to")
        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self._width))
        blueprint.set_attribute("image_size_y", str(self._height))
        self._camera = world.spawn_actor(
            blueprint,
            carla.Transform(
                carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-12.0)
            ),
            attach_to=vehicle,
        )
        self._camera.listen(self._queue.put)

    def close(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
            self._camera = None


def _cite(value: object, artifact: str, field_path: str) -> str:
    """One machine-checked citation in the ``docs/CLAIMS.md`` convention.

    The quote is ``str(value)`` of the artifact's own (already rounded) JSON
    value, so the checker's decimal-count comparison matches by construction —
    quoting a re-rounded number here would be the drift the checker exists to
    catch."""
    return f'[{value}]({artifact} "claim:{field_path}")'


def render_snippet(report: dict, *, source: str) -> str:
    """Render a ``demo_capture`` report's write-up: the README paste block.

    For a CARLA capture the write-up carries a fenced Markdown block to paste
    into the README verbatim — the embed plus a caption naming the Policy, the
    seed, and the Episode's own score, every number cited against ``source``
    and the media files cited as prose claims. The fence keeps the citations
    inert in this file (their paths are README-relative, not write-up
    relative) while the claim checker enforces them the moment they land in
    the README.

    A pipeline-only capture gets no paste block at all: a kinematic clip in
    the README would present a physics-free render as the demo, so the
    write-up says exactly that instead.
    """
    directory = Path(source).parent
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
    json_path = source
    lines += [
        "Paste the block below into the README's demo slot (the issue #27 "
        "placeholder), replacing it. Paths are README-relative, so the report "
        "and media must live where this capture wrote them, committed to the "
        "repo. The claim checker then verifies every number and the media "
        "files' existence in CI.",
        "",
        "```markdown",
        f'<img src="{gif}" alt="{policy["model_version"]} driving a seeded CARLA '
        f'Episode in {episode["town"]}">',
        "",
        f"*A seeded CARLA Episode — {episode['town']}, {episode['weather']}, seed "
        f"{_cite(episode['seed'], json_path, 'episode.seed')} — driven by this "
        f"project's `{policy['model_version']}` with {policy['perception']} "
        "perception (the ablation's baseline arm): driving score "
        f"{_cite(score['driving_score'], json_path, 'score.driving_score')}, "
        "completing "
        f"{_cite(score['route_completion'], json_path, 'score.route_completion')} "
        "of its "
        f"{_cite(episode['route_length_m'], json_path, 'episode.route_length_m')} m "
        "route. The GIF shows the first "
        f"{_cite(media['gif_covers_seconds'], json_path, 'media.gif_covers_seconds')} s "
        f"in real time; [the full clip]({video} \"claim:prose\") and "
        f"[the capture report]({json_path} \"claim:prose\") sit beside "
        f"[it]({gif} \"claim:prose\"). Recorded by `python -m pathfinder.demo "
        "--backend carla`.*",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record one benchmarked Episode as the README demo clip (issue #27). "
            "Run from the repo root so the generated paste block's paths are "
            "README-relative."
        )
    )
    parser.add_argument(
        "--backend",
        choices=["kinematic", "carla"],
        default="kinematic",
        # 'auto' is deliberately not offered, same as the ablation CLI: only a
        # named carla run may become the README demo.
        help="simulator backend; only a carla capture may be embedded as the demo",
    )
    parser.add_argument(
        "--episode-index", type=int, default=0,
        help="which Episode of the benchmark suite to replay (default: ep-0000)",
    )
    parser.add_argument("--route-length-m", type=float, default=400.0)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument(
        "--gif-seconds", type=float, default=16.0,
        help="how much of the Episode the README GIF covers; the MP4 gets all of it",
    )
    parser.add_argument(
        "--gif-stride", type=int, default=3,
        help="keep every Nth frame in the GIF (playback stays real time)",
    )
    parser.add_argument("--gif-width", type=int, default=480, help="GIF width in pixels")
    parser.add_argument(
        "--scale", type=int, default=None,
        help="integer upscale of recorded frames; default 4 on kinematic "
        "(its native render is 200x88), 1 on carla",
    )
    parser.add_argument(
        "--camera-width", type=int, default=960, help="chase camera width (carla only)"
    )
    parser.add_argument(
        "--camera-height", type=int, default=540, help="chase camera height (carla only)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="report path; defaults to results/demo/<backend>_demo.json, with "
        "the media files landing beside it",
    )
    args = parser.parse_args(argv)

    from pathfinder.sim.carla_backend import build_simulator

    spec = suite_episode(
        args.episode_index,
        route_length_m=args.route_length_m,
        base_seed=args.base_seed,
        max_steps=args.max_steps,
    )
    fps = 1.0 / spec.delta_seconds
    output = args.output or Path("results/demo") / f"{args.backend}_demo.json"
    artifact = reporting.ReportArtifact(output, label_key="policy")
    recorder = DemoRecorder(
        output.with_suffix(""),
        fps=fps,
        gif_stride=args.gif_stride,
        gif_max_frames=max(1, math.ceil(args.gif_seconds * fps / args.gif_stride)),
        gif_width=args.gif_width,
        scale=args.scale or (4 if args.backend == "kinematic" else 1),
    )
    simulator_kwargs = {"render": True} if args.backend == "kinematic" else {}

    chase = None
    with build_simulator(args.backend, **simulator_kwargs) as simulator:
        try:
            if args.backend == "carla":
                chase = ChaseCamera(
                    simulator, width=args.camera_width, height=args.camera_height
                )
            report = capture_demo(
                simulator,
                spec,
                recorder=recorder,
                frame_source=chase.source if chase is not None else None,
            )
        finally:
            if chase is not None:
                chase.close()

    writeup = artifact.finish(report, render_snippet)

    score = report["score"]
    print(
        f"episode {spec.episode_id} on {report['backend']} ({report['scope']}): "
        f"driving score {score['driving_score']}, route completion "
        f"{score['route_completion']}, {report['media']['frames']} frames recorded"
    )
    if report["scope"] == reporting.SCOPE_DRIVING_QUALITY:
        block = writeup.read_text(encoding="utf-8").split("```markdown\n")[1].split("```")[0]
        print("\nPaste into the README demo slot (replacing the issue #27 placeholder):\n")
        print(block)
    reporting.print_cli_tail(report, output, writeup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
