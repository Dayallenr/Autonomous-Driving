"""
Tests for scripts/provision_sqs.py: the LocalStack stand-in for issue #18's
targeted Terraform apply. The #22 rehearsal needs the queue pair to exist with
exactly the shape terraform/sqs.tf would create — same names, same visibility
timeout, same redrive policy — so the rehearsal's queue semantics are the real
run's, and the only thing the real sitting swaps is where the queues came
from. moto stands in for the SQS API, as in test_cloud_aws.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import provision_sqs  # noqa: E402 - needs scripts/ on sys.path first
from provision_sqs import main, provision  # noqa: E402

REGION = "us-east-1"
ENDPOINT = "http://localhost:4566"


@pytest.fixture
def aws(monkeypatch):
    # moto only intercepts default-endpoint clients, so route the script's
    # client factory around its explicit LocalStack endpoint. The endpoint
    # plumbing itself is proven live by the #22 rehearsal.
    monkeypatch.setattr(
        provision_sqs,
        "_sqs_client",
        lambda endpoint_url, region: boto3.client("sqs", region_name=region),
    )
    with mock_aws():
        yield


def _attributes(queue_url: str) -> dict:
    client = boto3.client("sqs", region_name=REGION)
    return client.get_queue_attributes(QueueUrl=queue_url, AttributeNames=["All"])[
        "Attributes"
    ]


def test_provision_mirrors_terraform_queue_shape(aws):
    queue_url, dlq_url = provision(
        "pathfinder-episodes", endpoint_url=ENDPOINT, region=REGION
    )

    assert queue_url.endswith("pathfinder-episodes")
    assert dlq_url.endswith("pathfinder-episodes-dlq")

    main_attributes = _attributes(queue_url)
    # terraform/sqs.tf: var.sqs_visibility_timeout_seconds default, and the
    # redrive policy targeting the DLQ after var.sqs_max_receive_count.
    assert main_attributes["VisibilityTimeout"] == "30"
    redrive = json.loads(main_attributes["RedrivePolicy"])
    assert redrive["maxReceiveCount"] == 3
    assert redrive["deadLetterTargetArn"] == _attributes(dlq_url)["QueueArn"]

    # terraform/sqs.tf: 14-day retention so a poison message is never
    # silently lost.
    assert _attributes(dlq_url)["MessageRetentionPeriod"] == "1209600"


def test_provision_is_idempotent(aws):
    first = provision("pathfinder-episodes", endpoint_url=ENDPOINT, region=REGION)
    second = provision("pathfinder-episodes", endpoint_url=ENDPOINT, region=REGION)
    assert first == second


def test_provision_refuses_a_real_aws_endpoint(aws):
    # Real queues come from Terraform (#18); this script must not be able to
    # create resources on real AWS even by accident.
    with pytest.raises(SystemExit, match="Terraform"):
        provision(
            "pathfinder-episodes",
            endpoint_url="https://sqs.us-east-1.amazonaws.com",
            region=REGION,
        )


def test_cli_requires_an_endpoint(aws):
    with pytest.raises(SystemExit):
        main([])


def test_cli_prints_urls_as_assignments(aws, capsys):
    assert main(["--endpoint-url", ENDPOINT]) == 0
    out = capsys.readouterr().out
    lines = dict(
        line.split("=", 1) for line in out.strip().splitlines() if "=" in line
    )
    assert lines["QUEUE_URL"].endswith("pathfinder-episodes")
    assert lines["DLQ_URL"].endswith("pathfinder-episodes-dlq")
