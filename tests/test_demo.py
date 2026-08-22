"""
The demo capture (issue #27): one seeded Episode recorded to media, with a
sidecar report the README caption can cite through the claim checker.

Everything here runs on the kinematic backend, so it proves the *mechanism* —
media encoding, report shape, self-labelling, snippet citations — on a machine
that cannot run CARLA. Deliberately absent: any assertion about what a CARLA
clip looks like; only the chase-camera spawn is CARLA-bound, and it stays as
thin as the backend's own sensor code.
"""
from __future__ import annotations

import numpy as np
import pytest

from pathfinder.demo import DemoRecorder, capture_demo
from pathfinder.demo_writeup import extract_paste_block, render_writeup
from pathfinder.sim import EpisodeSpec, KinematicSimulator


def _frames(count: int, *, height: int = 40, width: int = 64) -> list[np.ndarray]:
    """Synthetic RGB frames with a moving stripe, so encoded output has real
    frame-to-frame change rather than a constant image a codec could collapse."""
    frames = []
    for index in range(count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, (index * 2) % width] = (255, 128, 0)
        frames.append(frame)
    return frames


class TestDemoRecorder:
    def test_writes_video_with_every_frame_and_a_subsampled_gif(self, tmp_path):
        recorder = DemoRecorder(
            tmp_path / "demo", fps=20.0, gif_stride=3, gif_max_frames=8, gif_width=32
        )
        for frame in _frames(30):
            recorder.add(frame)
        media = recorder.close()

        video = tmp_path / media["video"]
        gif = tmp_path / media["gif"]
        assert video.is_file() and video.suffix == ".mp4"
        assert gif.is_file() and gif.suffix == ".gif"
        assert media["frames"] == 30
        # Every 3rd frame, capped at 8: frames 0, 3, 6, ... 21.
        assert media["gif_frames"] == 8

        import cv2

        capture = cv2.VideoCapture(str(video))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
        finally:
            capture.release()

        from PIL import Image

        with Image.open(gif) as image:
            assert image.n_frames == 8
            assert image.width == 32

    def test_upscales_small_frames_by_the_integer_scale(self, tmp_path):
        recorder = DemoRecorder(
            tmp_path / "demo", fps=20.0, gif_stride=1, gif_max_frames=5,
            gif_width=64, scale=4,
        )
        for frame in _frames(5):
            recorder.add(frame)
        media = recorder.close()

        assert media["width"] == 64 * 4
        assert media["height"] == 40 * 4

    def test_no_frames_writes_no_files(self, tmp_path):
        recorder = DemoRecorder(
            tmp_path / "demo", fps=20.0, gif_stride=3, gif_max_frames=8, gif_width=32
        )
        media = recorder.close()

        assert media["frames"] == 0
        assert list(tmp_path.iterdir()) == []


def _spec(**overrides) -> EpisodeSpec:
    defaults = dict(episode_id="demo-00", route_length_m=80.0, max_steps=30, seed=1000)
    return EpisodeSpec(**{**defaults, **overrides})


def _recorder(tmp_path, **overrides) -> DemoRecorder:
    defaults = dict(fps=20.0, gif_stride=3, gif_max_frames=40, gif_width=100, scale=2)
    return DemoRecorder(tmp_path / "kinematic_demo", **{**defaults, **overrides})


@pytest.fixture(scope="module")
def captured(tmp_path_factory) -> tuple:
    tmp_path = tmp_path_factory.mktemp("capture")
    with KinematicSimulator(render=True) as simulator:
        report = capture_demo(simulator, _spec(), recorder=_recorder(tmp_path))
    return report, tmp_path


class TestCaptureDemo:
    def test_report_identifies_the_run_that_produced_the_clip(self, captured):
        report, _ = captured
        assert report["kind"] == "demo_capture"
        assert report["backend"] == "kinematic"
        assert report["episode"]["seed"] == 1000
        assert report["score"]["episode_id"] == "demo-00"

    def test_policy_attribution_is_observed_from_telemetry(self, captured):
        """The caption must name what actually drove: the controller's registry
        name, and the perception observed on the Episode's own frames."""
        report, _ = captured
        assert report["policy"]["model_version"] == "pure_pursuit"
        assert report["policy"]["perception"] == "privileged"

    def test_kinematic_capture_labels_itself_pipeline_only(self, captured):
        report, _ = captured
        assert report["scope"] == "pipeline-only"
        assert "not driving quality" in report["scope_note"]

    def test_media_covers_every_frame_the_episode_scored(self, captured):
        report, tmp_path = captured
        assert report["media"]["frames"] == report["score"]["frames"] > 0
        assert (tmp_path / report["media"]["video"]).is_file()
        assert (tmp_path / report["media"]["gif"]).is_file()

    def test_a_capture_with_no_frames_refuses_to_report(self, tmp_path):
        """A backend that renders nothing cannot produce a demo; silence here
        would land a report whose caption cites media that does not exist."""
        with KinematicSimulator(render=False) as simulator:
            with pytest.raises(RuntimeError, match="no frames"):
                capture_demo(simulator, _spec(), recorder=_recorder(tmp_path))


def _carla_shaped(report: dict) -> dict:
    """The kinematic capture re-labelled as the CARLA run the snippet is for.

    Only the backend and its derived scope change; every number stays the
    kinematic run's own, so the snippet test exercises real values without a
    CARLA server. The writeup consumes a report dict — this is its contract.
    """
    from pathfinder.demo import _scope

    scope, scope_note = _scope("carla")
    return {**report, "backend": "carla", "scope": scope, "scope_note": scope_note}


class TestRenderSnippet:
    def test_carla_snippet_carries_a_paste_block_with_cited_caption(self, captured):
        report, _ = captured
        snippet = render_writeup(
            _carla_shaped(report), source="results/demo/carla_demo.json"
        )

        gif = "results/demo/" + report["media"]["gif"]
        video = "results/demo/" + report["media"]["video"]
        assert f'<img src="{gif}"' in snippet
        # The caption's numbers are cited in the checker convention, against
        # the sidecar this capture wrote.
        assert '"claim:score.driving_score"' in snippet
        assert '"claim:episode.seed"' in snippet
        # The media files are prose claims, so the checker enforces that the
        # embedded clip actually exists in the repo.
        assert f'({gif} "claim:prose")' in snippet
        assert f'({video} "claim:prose")' in snippet
        assert "pure_pursuit" in snippet
        assert report["episode"]["town"] in snippet

    def test_pipeline_only_snippet_has_no_embed_and_says_so(self, captured):
        report, _ = captured
        snippet = render_writeup(report, source="results/demo/kinematic_demo.json")

        assert "never the README demo" in snippet
        assert "<img" not in snippet
        assert extract_paste_block(snippet) is None

    def test_windows_source_path_still_yields_forward_slash_citations(self, captured):
        """The capture runs in PowerShell, where the default output path
        stringifies with backslashes. GitHub cannot resolve those and the
        claim checker on Linux CI cannot find the artifacts, so the paste
        block must come out forward-slashed regardless of platform."""
        report, _ = captured
        snippet = render_writeup(
            _carla_shaped(report), source="results\\demo\\carla_demo.json"
        )

        assert "\\" not in snippet
        assert '<img src="results/demo/' in snippet
        assert '(results/demo/carla_demo.json "claim:score.driving_score")' in snippet

    def test_paste_block_citations_pass_the_claim_checker(self, captured, tmp_path):
        """The end the snippet exists for: pasted into an opted-in document at
        the repo root, every citation resolves against the artifacts the
        capture wrote — including the media files' existence."""
        from pathfinder.claims import check_document

        report, media_dir = captured
        root = tmp_path / "repo"
        demo_dir = root / "results" / "demo"
        demo_dir.mkdir(parents=True)
        import json
        import shutil

        source = "results/demo/kinematic_demo.json"
        (root / source).write_text(json.dumps(report), encoding="utf-8")
        for key in ("video", "gif"):
            shutil.copy(media_dir / report["media"][key], demo_dir)

        snippet = render_writeup(_carla_shaped(report), source=source)
        fenced = extract_paste_block(snippet)
        assert fenced is not None
        readme = root / "README.md"
        readme.write_text("<!-- claims: checked -->\n\n" + fenced, encoding="utf-8")

        result = check_document(readme)
        assert result.failures == []
        assert result.machine_checked >= 3
        assert result.prose_audited >= 2


def test_suite_episode_replays_the_ablation_suite_exactly():
    """The demo must record a benchmarked Episode, not a lookalike: the spec
    at every index equals the one the real CARLA ablation recorded in its
    artifact — the independent source of truth for what was benchmarked."""
    import json
    from pathlib import Path

    from pathfinder.demo import suite_episode

    recorded = json.loads(
        Path("results/ablation/carla_report.json").read_text(encoding="utf-8")
    )["episodes"]
    assert suite_episode(0).to_dict() == recorded[0]
    assert suite_episode(2).to_dict() == recorded[2]


class TestCli:
    def test_kinematic_end_to_end_lands_report_writeup_and_media(self, tmp_path, capsys):
        from pathfinder.demo import main

        output = tmp_path / "kinematic_demo.json"
        assert main(
            [
                "--backend", "kinematic",
                "--route-length-m", "80",
                "--max-steps", "30",
                "--output", str(output),
            ]
        ) == 0

        assert output.is_file()
        assert output.with_suffix(".md").is_file()
        assert output.with_suffix(".mp4").is_file()
        assert output.with_suffix(".gif").is_file()
        assert not output.with_suffix(".partial.jsonl").exists()
        assert "pipeline-only" in capsys.readouterr().out

    def test_writeup_regenerates_from_the_report_alone(self, tmp_path, capsys):
        """Fixing snippet wording must never require re-driving a ten-minute
        CARLA Episode — the write-up regenerates from the JSON, like every
        other report kind's."""
        from pathfinder.demo import main
        from pathfinder.demo_writeup import main as writeup_main

        output = tmp_path / "kinematic_demo.json"
        main(
            [
                "--backend", "kinematic",
                "--route-length-m", "80",
                "--max-steps", "30",
                "--output", str(output),
            ]
        )
        original = output.with_suffix(".md").read_text(encoding="utf-8")
        output.with_suffix(".md").unlink()

        assert writeup_main([str(output)]) == 0
        assert output.with_suffix(".md").read_text(encoding="utf-8") == original
