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

from pathfinder.cloud import cli as cloud_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive telemetry stream records to Parquet")
    # optional=False: draining the stream is this script's whole job, so a
    # 'none' telemetry backend would be meaningless here.
    cloud_cli.add_telemetry_args(parser, backend_default="kinesis", optional=False)
    cloud_cli.add_object_store_args(parser, backend_default="s3")

    parser.add_argument("--limit", type=int, default=100_000, help="Max records to drain")
    args = parser.parse_args()

    stream = cloud_cli.stream_from_args(args)
    warehouse = cloud_cli.warehouse_from_args(args)

    keys = warehouse.drain_stream(stream, limit=args.limit)

    print(f"wrote {warehouse.rows_written} row(s) across {len(keys)} Parquet file(s):")
    for key in keys:
        print(f"  {warehouse.store.uri_for(key)}")


if __name__ == "__main__":
    main()
