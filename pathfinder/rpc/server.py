"""
gRPC server bootstrap for the coordinator.

This is deliberately thin: all the interesting behaviour (worker liveness,
dedup, cooperative stop) lives in ``CoordinatorService``. This module's only
job is binding it to a real socket and shutting down cleanly, which is exactly
the part that was still missing — the service class existed and was tested in
isolation, but nothing ever put it on the network.

Run directly:
    python -m pathfinder.rpc.server --port 50051 --episodes-total 0

``--episodes-total 0`` means "unbounded" — the server (and ``WatchRun``) never
declares the run complete on its own; use ``request_stop`` or SIGTERM instead.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal

import grpc

from pathfinder.rpc.coordinator import DEFAULT_HEARTBEAT_SECONDS, CoordinatorService
from pathfinder.rpc.generated import pathfinder_pb2_grpc

logger = logging.getLogger(__name__)

__all__ = ["serve"]


async def serve(
    *,
    port: int = 50051,
    episodes_total: int = 0,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    run_id: str | None = None,
) -> None:
    """Start the coordinator gRPC server and block until it's told to stop.

    Listens for SIGTERM/SIGINT and shuts down gracefully (finishes in-flight
    RPCs rather than dropping them), which matters here specifically because
    ``WatchRun`` holds a long-lived streaming RPC open — a hard stop would cut
    a dashboard off mid-stream instead of letting it see the server go away.
    """
    service = CoordinatorService(
        episodes_total=episodes_total, heartbeat_seconds=heartbeat_seconds, run_id=run_id
    )
    server = grpc.aio.server()
    pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
    bind_address = f"[::]:{port}"
    server.add_insecure_port(bind_address)

    await server.start()
    logger.info("coordinator listening on %s (run_id=%s)", bind_address, service.run.run_id)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _handle_signal() -> None:
        logger.info("shutdown signal received")
        service.request_stop("server shutting down")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # add_signal_handler is unix-only; Windows falls back to
            # KeyboardInterrupt propagating out of server.wait_for_termination().
            pass

    await stop_event.wait()
    await server.stop(grace=5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="PathFinder coordinator gRPC server")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--episodes-total",
        type=int,
        default=0,
        help="Expected episode count for this run; 0 means unbounded/unknown.",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=DEFAULT_HEARTBEAT_SECONDS)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(
        serve(
            port=args.port,
            episodes_total=args.episodes_total,
            heartbeat_seconds=args.heartbeat_seconds,
            run_id=args.run_id,
        )
    )


if __name__ == "__main__":
    main()
