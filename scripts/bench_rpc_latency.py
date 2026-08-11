"""
Measure real coordinator gRPC round-trip latency and write a report.

This exists to replace an invented number: the project previously claimed
"10ms round-trip latency" for the gRPC service without ever having bound a
server to a port or made a single real call. This script makes an actual
client talk to an actual server over loopback TCP and reports what really
happens, per RPC method, so the number in results/rpc/latency_report.json is
one this repo can reproduce on demand.

Usage:
    python scripts/bench_rpc_latency.py --calls 500
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc

from pathfinder.metrics.driving_score import EpisodeScore
from pathfinder.rpc.client import CoordinatorClient, episode_score_to_proto
from pathfinder.rpc.coordinator import CoordinatorService
from pathfinder.rpc.generated import pathfinder_pb2_grpc


def _percentile(samples_ms: list[float], pct: float) -> float:
    ordered = sorted(samples_ms)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _summarize(name: str, samples_ms: list[float]) -> dict:
    return {
        "rpc": name,
        "calls": len(samples_ms),
        "mean_ms": round(statistics.mean(samples_ms), 3),
        "p50_ms": round(_percentile(samples_ms, 50), 3),
        "p95_ms": round(_percentile(samples_ms, 95), 3),
        "p99_ms": round(_percentile(samples_ms, 99), 3),
        "max_ms": round(max(samples_ms), 3),
    }


async def _bench(calls: int) -> dict:
    service = CoordinatorService(episodes_total=0)
    server = grpc.aio.server()
    pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()

    client = CoordinatorClient.connect(f"127.0.0.1:{port}")
    results: dict[str, list[float]] = {
        "RegisterWorker": [],
        "Heartbeat": [],
        "SubmitResult": [],
        "GetRunStatus": [],
    }

    try:
        # RegisterWorker: one call per synthetic worker, since re-registering
        # the same worker_id isn't the steady-state path this RPC is used for.
        for index in range(calls):
            started = time.perf_counter()
            await client.register_worker(f"bench-worker-{index}", simulator_backend="kinematic")
            results["RegisterWorker"].append((time.perf_counter() - started) * 1000.0)

        for index in range(calls):
            started = time.perf_counter()
            await client.heartbeat_once(
                "bench-worker-0", current_episode_id=f"ep-{index}", episode_progress=0.5
            )
            results["Heartbeat"].append((time.perf_counter() - started) * 1000.0)

        for index in range(calls):
            score = EpisodeScore(
                episode_id=f"bench-ep-{index}",
                route_completion=0.9,
                infraction_penalty=1.0,
                driving_score=90.0,
                frames=100,
                duration_seconds=5.0,
                distance_travelled_m=360.0,
                route_length_m=400.0,
            )
            proto_result = episode_score_to_proto(
                score, worker_id="bench-worker-0", simulator_backend="kinematic"
            )
            started = time.perf_counter()
            await client.submit_result(proto_result)
            results["SubmitResult"].append((time.perf_counter() - started) * 1000.0)

        for _ in range(calls):
            started = time.perf_counter()
            await client.get_run_status(service.run.run_id)
            results["GetRunStatus"].append((time.perf_counter() - started) * 1000.0)
    finally:
        await client.close()
        await server.stop(None)

    return {
        "calls_per_rpc": calls,
        "transport": "grpc.aio over loopback TCP (127.0.0.1), insecure channel",
        "results": [_summarize(name, samples) for name, samples in results.items()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark coordinator gRPC round-trip latency")
    parser.add_argument("--calls", type=int, default=500, help="Calls per RPC method")
    parser.add_argument(
        "--output", type=Path, default=Path("results/rpc/latency_report.json")
    )
    args = parser.parse_args()

    report = asyncio.run(_bench(args.calls))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {args.output}")
    for entry in report["results"]:
        print(
            f"  {entry['rpc']:16} p50={entry['p50_ms']:6.3f}ms  "
            f"p99={entry['p99_ms']:6.3f}ms  mean={entry['mean_ms']:6.3f}ms  "
            f"n={entry['calls']}"
        )


if __name__ == "__main__":
    main()
