"""
Tests for scripts/sqs_apply_wizard.sh: issue #18's interactive wizard for the
one real-AWS provisioning step. The wizard itself is a human-driven bash
script, so what is pinned here is the part that must never drift: the set of
Terraform resource addresses it is allowed to touch. Everything else in
terraform/ stays validated-never-applied, and these tests fail if the wizard
grows a target outside the budget alarm + episode-queue family, or if the
addresses it targets stop existing in the Terraform configuration.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WIZARD = REPO_ROOT / "scripts" / "sqs_apply_wizard.sh"
TF_DIR = REPO_ROOT / "terraform"

# The wizard's entire permitted footprint: the ground-rules $1 budget alarm,
# the episode queue pair, and the DLQ's redrive-allow policy attachment (an
# SQS attribute on the queues it targets, not a new billable resource).
ALLOWED_TARGETS = {
    "aws_budgets_budget.zero_spend",
    "aws_sqs_queue.episodes",
    "aws_sqs_queue.dead_letter",
    "aws_sqs_queue_redrive_allow_policy.dead_letter",
}


def _wizard_text() -> str:
    return WIZARD.read_text(encoding="utf-8")


def _terraform_resource_addresses() -> set[str]:
    addresses = set()
    for tf_file in TF_DIR.glob("*.tf"):
        text = tf_file.read_text(encoding="utf-8")
        for match in re.finditer(r'^resource\s+"([^"]+)"\s+"([^"]+)"', text, re.M):
            addresses.add(f"{match.group(1)}.{match.group(2)}")
    return addresses


def _wizard_targets() -> set[str]:
    # Every Terraform resource address the wizard mentions anywhere — as a
    # -target literal or via its BUDGET_TARGET/SQS_TARGETS variables — is
    # part of its footprint.
    return set(re.findall(r"\baws_[a-z0-9_]+\.[a-z0-9_]+\b", _wizard_text()))


def test_wizard_is_valid_bash():
    subprocess.run(["bash", "-n", str(WIZARD)], check=True)


def test_wizard_is_executable():
    assert WIZARD.stat().st_mode & 0o111, "wizard must be chmod +x"


def test_wizard_targets_exactly_the_budget_and_queue_family():
    # Equality, not subset: a target outside this set would weaken the
    # never-applied status of EKS/Kinesis/S3/ECR/KMS/IAM (#18 acceptance
    # criterion), and a missing one means the wizard no longer provisions
    # what the runbook expects.
    assert _wizard_targets() == ALLOWED_TARGETS


def test_wizard_targets_exist_in_terraform():
    missing = _wizard_targets() - _terraform_resource_addresses()
    assert not missing, f"wizard targets not found in terraform/: {sorted(missing)}"


def test_wizard_verifies_the_budget_alarm_via_the_aws_api():
    # The acceptance criterion is "a budget alarm exists before anything is
    # applied; the wizard refuses to continue without it" — existence must be
    # checked against the live Budgets API, not inferred from Terraform state.
    assert "describe-budget" in _wizard_text()


def test_budget_alarm_is_the_ground_rules_one_dollar_monthly():
    budget = (TF_DIR / "budget.tf").read_text(encoding="utf-8")
    assert re.search(r'limit_amount\s*=\s*"1\.0"', budget)
    assert re.search(r'limit_unit\s*=\s*"USD"', budget)
    assert re.search(r'time_unit\s*=\s*"MONTHLY"', budget)
    # At least one ACTUAL-spend notification wired to the alarm email.
    assert re.search(r'notification_type\s*=\s*"ACTUAL"', budget)
    assert "var.budget_notification_email" in budget


def test_budget_email_variable_is_required_non_default():
    # Like github_repository: no default, so the alarm can never be created
    # pointing at nobody.
    variables = (TF_DIR / "variables.tf").read_text(encoding="utf-8")
    match = re.search(
        r'variable\s+"budget_notification_email"\s*\{(.*?)\n\}',
        variables,
        re.S,
    )
    assert match, "budget_notification_email must be declared in variables.tf"
    assert not re.search(r"^\s*default\s*=", match.group(1), re.M)
