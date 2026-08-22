"""
Provision the episode queue pair on LocalStack, shaped exactly like
terraform/sqs.tf — the #22 rehearsal's stand-in for issue #18's targeted
Terraform apply.

The rehearsal must exercise the same queue semantics as the real run: same
names (`pathfinder-episodes` + `pathfinder-episodes-dlq`), same visibility
timeout, same redrive policy. This script creates that pair against a
LocalStack endpoint and prints the two URLs as NAME=value lines, ready to
paste into the runbook's configuration block.

It deliberately cannot provision real AWS: --endpoint-url is required and an
amazonaws.com endpoint is refused. Real queues are Terraform's job (#18) —
that keeps "written, never applied" true for everything but the one wizard
that applies the queues on purpose.

Usage:
    python scripts/provision_sqs.py --endpoint-url http://localhost:4566
"""
from __future__ import annotations

import argparse
import json

# Mirrors terraform/sqs.tf (and its variable defaults). Change these there
# first; this script only follows.
VISIBILITY_TIMEOUT_SECONDS = 30
MAX_RECEIVE_COUNT = 3
DLQ_RETENTION_SECONDS = 1_209_600  # 14 days — the maximum


def _sqs_client(endpoint_url: str, region: str):
    # Its own seam because moto can only intercept default-endpoint clients;
    # tests patch this, and the LocalStack rehearsal proves the endpoint
    # plumbing live.
    import boto3

    return boto3.client("sqs", endpoint_url=endpoint_url, region_name=region)


def provision(
    name: str, *, endpoint_url: str, region: str = "us-east-1"
) -> tuple[str, str]:
    """Create-or-resolve the main queue and its DLQ; return their URLs.

    ``create_queue`` with identical attributes is idempotent, so re-running
    against queues this script already created just returns their URLs.

    Raises:
        SystemExit: On an amazonaws.com endpoint — real queues come from
            Terraform (#18), never from this script.
    """
    if "amazonaws.com" in endpoint_url:
        raise SystemExit(
            "refusing to provision real AWS: the real queue pair comes from "
            "issue #18's targeted Terraform apply, not this script. Point "
            "--endpoint-url at LocalStack."
        )
    client = _sqs_client(endpoint_url, region)

    dlq_url = client.create_queue(
        QueueName=f"{name}-dlq",
        Attributes={
            "MessageRetentionPeriod": str(DLQ_RETENTION_SECONDS),
            "SqsManagedSseEnabled": "true",
        },
    )["QueueUrl"]
    dlq_arn = client.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])[
        "Attributes"
    ]["QueueArn"]

    queue_url = client.create_queue(
        QueueName=name,
        Attributes={
            "VisibilityTimeout": str(VISIBILITY_TIMEOUT_SECONDS),
            "SqsManagedSseEnabled": "true",
            "RedrivePolicy": json.dumps(
                {"deadLetterTargetArn": dlq_arn, "maxReceiveCount": MAX_RECEIVE_COUNT}
            ),
        },
    )["QueueUrl"]
    return queue_url, dlq_url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create the episode queue + DLQ pair on LocalStack, mirroring "
            "terraform/sqs.tf. LocalStack only — real queues come from the "
            "#18 Terraform apply."
        )
    )
    parser.add_argument(
        "--endpoint-url", type=str, required=True,
        help="LocalStack SQS endpoint, e.g. http://localhost:4566",
    )
    parser.add_argument("--name", type=str, default="pathfinder-episodes")
    parser.add_argument("--region", type=str, default="us-east-1")
    args = parser.parse_args(argv)

    queue_url, dlq_url = provision(
        args.name, endpoint_url=args.endpoint_url, region=args.region
    )
    print(f"QUEUE_URL={queue_url}")
    print(f"DLQ_URL={dlq_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
