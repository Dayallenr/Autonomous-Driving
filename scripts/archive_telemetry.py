"""
Drain a telemetry stream into partitioned Parquet in an object store.

Runs offline, after a benchmark finishes — see pathfinder/cloud/warehouse.py's
docstring for why: the driving loop has a hard real-time budget, and a Parquet
encode inside it would show up as dropped frames. This script is the other
half of the pipeline scripts/run_worker.py's --telemetry-backend feeds:
workers push per-frame rows onto a stream (local JSONL or real Kinesis), and
this reads that stream back and writes it out as columnar Parquet, ready for
Athena/Redshift (see pathfinder.cloud.warehouse.athena_ddl/redshift_ddl).

Usage (LocalStack or real AWS — same code path, only --*-endpoint-url differs):
    python scripts/archive_telemetry.py \\
        --telemetry-backend kinesis --telemetry-stream-name pathfinder-telemetry \\
        --object-backend s3 --bucket pathfinder-telemetry-<account-id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathfinder.cloud.objects import build_store
from pathfinder.cloud.stream import build_stream
from pathfinder.cloud.warehouse import TelemetryWarehouse


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive telemetry stream records to Parquet")
    parser.add_argument("--telemetry-backend", choices=["local", "kinesis"], default="kinesis")
    parser.add_argument("--telemetry-stream-name", type=str, default="pathfinder-telemetry")
    parser.add_argument("--telemetry-local-root", type=str, default="./telemetry")
    parser.add_argument("--telemetry-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--telemetry-region", type=str, default="us-east-1")

    parser.add_argument("--object-backend", choices=["local", "s3"], default="s3")
    parser.add_argument("--bucket", type=str, default="pathfinder")
    parser.add_argument("--object-local-root", type=str, default="./s3")
    parser.add_argument("--object-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--object-region", type=str, default="us-east-1")
    parser.add_argument("--warehouse-prefix", type=str, default="telemetry")

    parser.add_argument("--limit", type=int, default=100_000, help="Max records to drain")
    args = parser.parse_args()

    stream = build_stream(
        args.telemetry_backend,
        local_root=args.telemetry_local_root,
        stream_name=args.telemetry_stream_name,
        endpoint_url=args.telemetry_endpoint_url,
        region=args.telemetry_region,
    )
    store = build_store(
        args.object_backend,
        local_root=args.object_local_root,
        bucket=args.bucket,
        endpoint_url=args.object_endpoint_url,
        region=args.object_region,
    )
    warehouse = TelemetryWarehouse(store, prefix=args.warehouse_prefix)

    keys = warehouse.drain_stream(stream, limit=args.limit)

    print(f"wrote {warehouse.rows_written} row(s) across {len(keys)} Parquet file(s):")
    for key in keys:
        print(f"  {store.uri_for(key)}")


if __name__ == "__main__":
    main()
