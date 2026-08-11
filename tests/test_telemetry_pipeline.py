"""
End-to-end test of the telemetry pipeline: a worker drives an episode,
streams per-frame telemetry to Kinesis, and scripts/archive_telemetry.py's
underlying drain logic writes it out as Parquet in S3 — the exact path Phase
7's real-AWS demo exercises, run here against moto instead of real AWS so it
costs nothing and needs no credentials.

This is what proves the Kinesis/S3 Terraform resources in terraform/ would
actually be used for something, rather than provisioned and left idle.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import boto3
import grpc
import pyarrow.parquet as pq
from moto import mock_aws

from pathfinder.cloud.objects import S3ObjectStore
from pathfinder.cloud.queue import LocalMessageQueue
from pathfinder.cloud.stream import KinesisTelemetryStream
from pathfinder.cloud.warehouse import TelemetryWarehouse
from pathfinder.rpc.coordinator import CoordinatorService
from pathfinder.rpc.generated import pathfinder_pb2_grpc
from pathfinder.sim.base import EpisodeSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_worker import run_worker  # noqa: E402 - needs scripts/ on sys.path first

REGION = "us-east-1"


def test_worker_telemetry_flows_through_kinesis_into_s3_parquet():
    with mock_aws():
        kinesis_client = boto3.client("kinesis", region_name=REGION)
        kinesis_client.create_stream(StreamName="pathfinder-telemetry-e2e", ShardCount=1)
        kinesis_client.get_waiter("stream_exists").wait(StreamName="pathfinder-telemetry-e2e")

        s3_client = boto3.client("s3", region_name=REGION)
        s3_client.create_bucket(Bucket="pathfinder-e2e-bucket")

        async def body():
            service = CoordinatorService(episodes_total=1)
            server = grpc.aio.server()
            pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()

            queue = LocalMessageQueue()
            spec = EpisodeSpec(
                episode_id="ep-telemetry-e2e", route_length_m=80.0, max_steps=200, seed=1
            )
            queue.send(spec.to_dict())

            telemetry_stream = KinesisTelemetryStream(
                "pathfinder-telemetry-e2e", region=REGION, batch_size=500
            )

            try:
                results = await run_worker(
                    worker_id="telemetry-e2e-worker",
                    coordinator_address=f"127.0.0.1:{port}",
                    queue=queue,
                    simulator_backend="kinematic",
                    idle_timeout_seconds=0.5,
                    receive_wait_seconds=0.2,
                    telemetry_stream=telemetry_stream,
                )
            finally:
                await server.stop(None)

            return results

        results = asyncio.run(body())
        assert len(results) == 1
        assert results[0].frames > 0

        # The worker's own KinesisTelemetryStream instance is closed by now
        # (run_worker's finally block flushes+closes it) — archival reads back
        # from a fresh stream handle, exactly as the standalone archive script
        # would after the worker process has already exited.
        archive_stream = KinesisTelemetryStream(
            "pathfinder-telemetry-e2e", region=REGION, batch_size=500
        )
        store = S3ObjectStore("pathfinder-e2e-bucket", region=REGION)
        warehouse = TelemetryWarehouse(store, prefix="telemetry")
        keys = warehouse.drain_stream(archive_stream)

        assert warehouse.rows_written == results[0].frames
        assert len(keys) >= 1

        for key in keys:
            data = store.get(key)
            table = pq.read_table(_bytesio(data))
            assert set(table.column("episode_id").to_pylist()) == {"ep-telemetry-e2e"}


def _bytesio(data: bytes):
    import io

    return io.BytesIO(data)
