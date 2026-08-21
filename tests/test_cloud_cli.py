"""
The shared CLI flag groups (#35): the queue, telemetry, and object-store
argparse blocks that scripts/run_worker.py, scripts/enqueue_episodes.py,
scripts/archive_telemetry.py, and pathfinder/distributed_run.py all mount
from pathfinder/cloud/cli.py.

These tests pin the flag names and defaults each CLI shipped with before the
consolidation — the #22 runbook depends on them not drifting — and the
build-from-args wiring, including the --queue-name -> ensure_queue fallback
that was previously copy-pasted per CLI.
"""
from __future__ import annotations

import argparse

import pytest

from pathfinder.cloud import cli
from pathfinder.cloud.objects import LocalObjectStore
from pathfinder.cloud.queue import LocalMessageQueue, SqsMessageQueue
from pathfinder.cloud.stream import LocalTelemetryStream


def parse(add_args, argv=(), **kwargs) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_args(parser, **kwargs)
    return parser.parse_args(list(argv))


class TestQueueArgs:
    def test_worker_parametrization_keeps_run_workers_defaults(self):
        args = parse(cli.add_queue_args, backend_default="local", queue_name_default="")
        assert args.queue_backend == "local"
        assert args.queue_url == ""
        assert args.queue_name == ""
        assert args.dead_letter_queue_url is None
        assert args.queue_endpoint_url is None
        assert args.queue_region == "us-east-1"

    def test_default_parametrization_matches_enqueuer_and_collector(self):
        args = parse(cli.add_queue_args)
        assert args.queue_backend == "sqs"
        assert args.queue_name == "pathfinder-episodes"

    def test_local_backend_builds_a_local_queue(self):
        args = parse(cli.add_queue_args, ["--queue-backend", "local"])
        assert isinstance(cli.queue_from_args(args), LocalMessageQueue)

    def test_explicit_url_skips_name_resolution(self, monkeypatch):
        def boom(*_args, **_kwargs):
            raise AssertionError("ensure_queue must not run when --queue-url is given")

        monkeypatch.setattr(cli, "ensure_queue", boom)
        args = parse(cli.add_queue_args, ["--queue-url", "https://sqs.example/q"])
        queue = cli.queue_from_args(args)
        assert isinstance(queue, SqsMessageQueue)
        assert queue.queue_url == "https://sqs.example/q"

    def test_name_resolves_through_ensure_queue(self, monkeypatch):
        seen = {}

        def fake_ensure(name, *, endpoint_url=None, region="us-east-1"):
            seen.update(name=name, endpoint_url=endpoint_url, region=region)
            return "https://resolved.example/q"

        monkeypatch.setattr(cli, "ensure_queue", fake_ensure)
        args = parse(cli.add_queue_args, ["--queue-endpoint-url", "http://localhost:4566"])
        queue = cli.queue_from_args(args)
        assert isinstance(queue, SqsMessageQueue)
        assert queue.queue_url == "https://resolved.example/q"
        assert seen == {
            "name": "pathfinder-episodes",
            "endpoint_url": "http://localhost:4566",
            "region": "us-east-1",
        }

    def test_sqs_with_neither_url_nor_name_exits_with_guidance(self):
        args = parse(
            cli.add_queue_args,
            ["--queue-backend", "sqs"],
            backend_default="local",
            queue_name_default="",
        )
        with pytest.raises(SystemExit, match="--queue-url or --queue-name"):
            cli.queue_from_args(args)


class TestTelemetryArgs:
    def test_default_parametrization_matches_worker_and_collector(self):
        args = parse(cli.add_telemetry_args)
        assert args.telemetry_backend == "none"
        assert args.telemetry_stream_name == "pathfinder-telemetry"
        assert args.telemetry_local_root == "./telemetry"
        assert args.telemetry_endpoint_url is None
        assert args.telemetry_region == "us-east-1"

    def test_archiver_parametrization_has_no_none_choice(self):
        parser = argparse.ArgumentParser()
        cli.add_telemetry_args(parser, backend_default="kinesis", optional=False)
        assert parser.parse_args([]).telemetry_backend == "kinesis"
        with pytest.raises(SystemExit):
            parser.parse_args(["--telemetry-backend", "none"])
        # The default help must not advertise the 'none' value the parser
        # just rejected.
        assert "none" not in parser.format_help()

    def test_none_builds_no_stream(self):
        assert cli.stream_from_args(parse(cli.add_telemetry_args)) is None

    def test_local_backend_builds_a_local_stream(self, tmp_path):
        args = parse(
            cli.add_telemetry_args,
            ["--telemetry-backend", "local", "--telemetry-local-root", str(tmp_path)],
        )
        assert isinstance(cli.stream_from_args(args), LocalTelemetryStream)


class TestObjectStoreArgs:
    def test_archiver_defaults(self):
        args = parse(cli.add_object_store_args, backend_default="s3")
        assert args.object_backend == "s3"
        assert args.bucket == "pathfinder"
        assert args.object_local_root == "./s3"
        assert args.object_endpoint_url is None
        assert args.object_region == "us-east-1"
        assert args.warehouse_prefix == "telemetry"

    def test_collector_defaults_to_local(self):
        args = parse(cli.add_object_store_args, backend_default="local")
        assert args.object_backend == "local"

    def test_store_and_warehouse_from_args(self, tmp_path):
        args = parse(
            cli.add_object_store_args,
            ["--object-local-root", str(tmp_path), "--warehouse-prefix", "frames"],
            backend_default="local",
        )
        store = cli.store_from_args(args)
        assert isinstance(store, LocalObjectStore)
        warehouse = cli.warehouse_from_args(args)
        assert isinstance(warehouse.store, LocalObjectStore)
        assert warehouse.prefix == "frames"
