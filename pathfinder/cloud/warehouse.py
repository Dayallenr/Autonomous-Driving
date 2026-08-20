"""
Telemetry warehouse: Kinesis records → partitioned Parquet → Redshift/Athena.

This is the offline-analysis half of the telemetry pipeline. The stream
(``cloud/stream.py``) is the online half; this module drains it into columnar
storage where regression questions are cheap to ask:

* "Did collision rate get worse between model v3 and v4 on Town05?"
* "Which route segments have the highest policy disagreement?"
* "What is the p99 policy latency by weather preset?"

None of those are answerable against a JSONL stream without a full scan, and all
of them are a partition-pruned column read against Parquet.

Two tables, because they have different grains
----------------------------------------------
``telemetry_frames`` is per-frame (20 Hz per worker — the high-volume table).
``episode_results`` is per-episode (one row per run — small, and the one
dashboards actually aggregate).

Keeping them separate means the common query ("driving score by model version")
reads a few thousand rows instead of tens of millions. Denormalizing episode
outcomes onto every frame would make that query 20,000x more expensive to save
one join.
"""
from __future__ import annotations

import io
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

__all__ = [
    "EPISODE_COLUMNS",
    "FRAME_COLUMNS",
    "Column",
    "TelemetryWarehouse",
    "athena_ddl",
    "redshift_ddl",
]


@dataclass(frozen=True)
class Column:
    name: str
    arrow: str
    redshift: str
    hive: str
    comment: str


#: Per-frame telemetry. Column order is stable: Redshift COPY from Parquet
#: matches on position when names are not supplied, so reordering silently
#: shifts data between columns. New columns go immediately before
#: ``event_date``, which must stay last — both DDL generators treat the final
#: column as the partition key.
FRAME_COLUMNS: tuple[Column, ...] = (
    Column("episode_id", "string", "VARCHAR(64)", "string", "joins to episode_results"),
    Column("frame_index", "int64", "BIGINT", "bigint", "monotonic within an episode"),
    Column("timestamp", "timestamp[us, tz=UTC]", "TIMESTAMPTZ", "timestamp", "wall clock"),
    Column("simulation_time", "double", "DOUBLE PRECISION", "double", "seconds of sim time"),
    Column("worker_id", "string", "VARCHAR(64)", "string", "which worker produced this"),
    Column("town", "string", "VARCHAR(32)", "string", "CARLA map"),
    Column("weather", "string", "VARCHAR(32)", "string", "weather preset"),
    Column("x", "double", "DOUBLE PRECISION", "double", "ego world x"),
    Column("y", "double", "DOUBLE PRECISION", "double", "ego world y"),
    Column("yaw_degrees", "double", "DOUBLE PRECISION", "double", "ego heading"),
    Column("speed_mps", "double", "DOUBLE PRECISION", "double", "ego speed"),
    Column("throttle", "double", "DOUBLE PRECISION", "double", "control output"),
    Column("brake", "double", "DOUBLE PRECISION", "double", "control output"),
    Column("steer", "double", "DOUBLE PRECISION", "double", "control output"),
    Column("command", "int64", "SMALLINT", "int", "CIL branch index 0..3"),
    Column("detections", "int64", "INTEGER", "int", "objects detected this frame"),
    Column("nearest_object_m", "double", "DOUBLE PRECISION", "double", "range to closest object"),
    Column("policy_latency_ms", "double", "DOUBLE PRECISION", "double", "policy step cost"),
    Column("perception_latency_ms", "double", "DOUBLE PRECISION", "double", "perception cost, distinct from policy cost"),
    Column("infraction", "string", "VARCHAR(32)", "string", "empty when none"),
    # Provenance, repeated on every frame. It duplicates across a whole episode,
    # which Parquet's dictionary encoding costs almost nothing for, and it is
    # what stops a frame from being read as evidence about a policy that did not
    # produce it — including the CARLA baseline, which is not project work.
    Column("model_version", "string", "VARCHAR(64)", "string", "policy that drove this frame"),
    Column("dataset_version", "string", "VARCHAR(64)", "string", "data that policy trained on"),
    Column("perception", "string", "VARCHAR(64)", "string", "perception that produced this frame's obstacle fields"),
    Column("event_date", "string", "VARCHAR(10)", "string", "YYYY-MM-DD partition key"),
)

#: Per-episode outcomes — the table dashboards aggregate.
EPISODE_COLUMNS: tuple[Column, ...] = (
    Column("episode_id", "string", "VARCHAR(64)", "string", "primary key"),
    Column("worker_id", "string", "VARCHAR(64)", "string", "which worker ran it"),
    Column("model_version", "string", "VARCHAR(64)", "string", "policy weights identifier"),
    Column("dataset_version", "string", "VARCHAR(64)", "string", "data those weights trained on"),
    Column("town", "string", "VARCHAR(32)", "string", "CARLA map"),
    Column("weather", "string", "VARCHAR(32)", "string", "weather preset"),
    Column("route_length_m", "double", "DOUBLE PRECISION", "double", "total route distance"),
    Column("route_completion", "double", "DOUBLE PRECISION", "double", "fraction in [0,1]"),
    Column("infraction_penalty", "double", "DOUBLE PRECISION", "double", "product of penalties"),
    Column("driving_score", "double", "DOUBLE PRECISION", "double", "completion x penalty, x100"),
    Column("collisions", "int64", "INTEGER", "int", "collision count"),
    Column("lane_invasions", "int64", "INTEGER", "int", "lane departure count"),
    Column("red_lights_run", "int64", "INTEGER", "int", "traffic light violations"),
    Column("duration_seconds", "double", "DOUBLE PRECISION", "double", "wall-clock episode time"),
    Column("frames", "int64", "BIGINT", "bigint", "frames simulated"),
    Column("mean_fps", "double", "DOUBLE PRECISION", "double", "throughput achieved"),
    Column("status", "string", "VARCHAR(32)", "string", "completed | failed | timeout"),
    Column("event_date", "string", "VARCHAR(10)", "string", "YYYY-MM-DD partition key"),
)

_ARROW_TYPES = {
    "string": "string",
    "double": "float64",
    "int64": "int64",
    "timestamp[us, tz=UTC]": "timestamp",
}


def _arrow_schema(columns: tuple[Column, ...]):
    import pyarrow as pa

    mapping = {
        "string": pa.string(),
        "double": pa.float64(),
        "int64": pa.int64(),
        "timestamp[us, tz=UTC]": pa.timestamp("us", tz="UTC"),
    }
    return pa.schema([pa.field(c.name, mapping[c.arrow]) for c in columns])


def redshift_ddl(*, schema: str = "analytics") -> str:
    """Redshift DDL for both tables.

    ``telemetry_frames`` uses ``DISTKEY(episode_id)`` so one episode's frames sit
    on one slice — every frame-level query is scoped to an episode, and
    co-location avoids a redistribution step. ``episode_results`` uses
    ``DISTSTYLE ALL`` because it is small and is the right side of nearly every
    join; replicating it to all slices removes the broadcast entirely.
    """
    blocks = []
    for table, columns, tail in (
        (
            "telemetry_frames",
            FRAME_COLUMNS,
            "DISTSTYLE KEY\nDISTKEY(episode_id)\nCOMPOUND SORTKEY(event_date, episode_id, frame_index);",
        ),
        (
            "episode_results",
            EPISODE_COLUMNS,
            "DISTSTYLE ALL\nCOMPOUND SORTKEY(event_date, model_version);",
        ),
    ):
        lines = []
        for index, column in enumerate(columns):
            separator = "," if index < len(columns) - 1 else ""
            declaration = f"{column.redshift}{separator}"
            # Comma before the comment: a trailing `-- text` swallows the rest
            # of the line, merging every following column into the comment.
            lines.append(f"    {column.name:24} {declaration:22}  -- {column.comment}")
        blocks.append(
            f"DROP TABLE IF EXISTS {schema}.{table};\n\n"
            f"CREATE TABLE {schema}.{table} (\n" + "\n".join(lines) + f"\n)\n{tail}"
        )

    return (
        "-- Generated by pathfinder.cloud.warehouse.redshift_ddl(); do not hand-edit.\n"
        f"CREATE SCHEMA IF NOT EXISTS {schema};\n\n" + "\n\n".join(blocks) + "\n\n"
        "-- Load a partition written by TelemetryWarehouse:\n"
        f"--   COPY {schema}.telemetry_frames\n"
        "--   FROM 's3://<bucket>/telemetry/frames/event_date=YYYY-MM-DD/'\n"
        "--   IAM_ROLE '<role-arn>' FORMAT AS PARQUET;\n"
    )


def athena_ddl(*, database: str = "pathfinder", location: str = "s3://pathfinder/telemetry/") -> str:
    """External-table DDL for querying the Parquet in place."""
    blocks = []
    for table, columns, subdir in (
        ("telemetry_frames", FRAME_COLUMNS, "frames"),
        ("episode_results", EPISODE_COLUMNS, "episodes"),
    ):
        partition = columns[-1]
        body = ",\n".join(
            f"    {column.name:24} {column.hive:12} COMMENT '{column.comment}'"
            for column in columns
            if column.name != partition.name
        )
        blocks.append(
            f"CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table} (\n{body}\n)\n"
            f"PARTITIONED BY ({partition.name} {partition.hive})\n"
            f"STORED AS PARQUET\n"
            f"LOCATION '{location.rstrip('/')}/{subdir}/'\n"
            f"TBLPROPERTIES ('parquet.compression'='SNAPPY');"
        )
    return (
        "-- Generated by pathfinder.cloud.warehouse.athena_ddl(); do not hand-edit.\n"
        f"CREATE DATABASE IF NOT EXISTS {database};\n\n" + "\n\n".join(blocks) + "\n\n"
        f"-- Register partitions after a write:\n"
        f"--   MSCK REPAIR TABLE {database}.telemetry_frames;\n"
    )


class TelemetryWarehouse:
    """Drains a telemetry stream into partitioned Parquet.

    Runs **offline**, after episodes finish, not inside the driving loop. The
    control loop has a hard real-time budget; a Parquet encode inside it would
    show up directly as dropped frames.
    """

    def __init__(self, store, *, prefix: str = "telemetry") -> None:
        self.store = store
        self.prefix = prefix.strip("/")
        self.files_written = 0
        self.rows_written = 0

    def _write(self, columns: tuple[Column, ...], rows: list[dict], subdir: str) -> list[str]:
        """Write rows as one Parquet file per date partition."""
        if not rows:
            return []

        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = _arrow_schema(columns)
        by_partition: dict[str, list[dict]] = {}
        for row in rows:
            by_partition.setdefault(row["event_date"], []).append(row)

        keys: list[str] = []
        for event_date, partition_rows in sorted(by_partition.items()):
            # Build column-wise in the schema's declared order; constructing
            # from a row list lets pyarrow infer order from the first row, which
            # silently reorders columns and breaks positional COPY.
            table = pa.Table.from_pydict(
                {
                    field.name: [row.get(field.name) for row in partition_rows]
                    for field in schema
                },
                schema=schema,
            )
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression="snappy")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            key = (
                f"{self.prefix}/{subdir}/event_date={event_date}/"
                f"part-{stamp}-{uuid.uuid4().hex[:8]}.parquet"
            )
            self.store.put(key, buffer.getvalue())
            keys.append(key)
            self.files_written += 1
            self.rows_written += len(partition_rows)
        return keys

    def write_frames(self, rows: list[dict]) -> list[str]:
        """Write per-frame telemetry rows."""
        return self._write(FRAME_COLUMNS, rows, "frames")

    def write_episodes(self, rows: list[dict]) -> list[str]:
        """Write per-episode result rows."""
        return self._write(EPISODE_COLUMNS, rows, "episodes")

    def drain_stream(self, stream, *, limit: int = 100_000) -> list[str]:
        """Read a telemetry stream and write its frames to Parquet.

        Records missing required columns are skipped with a warning rather than
        failing the whole drain: one malformed frame should not cost an entire
        benchmark run's telemetry.
        """
        records = stream.read(limit=limit)
        rows: list[dict] = []
        skipped = 0
        for record in records:
            data = dict(record.data)
            if "episode_id" not in data or "frame_index" not in data:
                skipped += 1
                continue
            data.setdefault("event_date", datetime.now(UTC).strftime("%Y-%m-%d"))
            # Timestamps cross the stream as ISO-8601 strings; the Arrow schema
            # declares a timestamp column, so parse them back on the way in.
            stamp = data.get("timestamp")
            if isinstance(stamp, str):
                try:
                    data["timestamp"] = datetime.fromisoformat(stamp)
                except ValueError:
                    data["timestamp"] = datetime.now(UTC)
            elif stamp is None:
                data["timestamp"] = datetime.now(UTC)
            rows.append(data)
        if skipped:
            logger.warning("skipped %d malformed telemetry record(s) during drain", skipped)
        return self.write_frames(rows)
