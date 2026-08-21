"""The claim checker (issue #19): documents held to their artifacts.

Fixture-driven — the convention is proven here before any real document opts
in. The convention itself is documented in docs/CLAIMS.md; these tests are the
executable version of that page's rules.
"""
from __future__ import annotations

import json
from pathlib import Path

from pathfinder.claims import CHECKED_MARKER, check_document, check_repo

REPO_ROOT = Path(__file__).resolve().parent.parent


def write_report(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_document_with_one_correct_numeric_claim_passes(tmp_path):
    write_report(tmp_path / "report.json", {"difference": {"driving_score": 16.34}})
    doc = tmp_path / "doc.md"
    doc.write_text(
        f"{CHECKED_MARKER}\n\nPerception cost "
        '[16.34](report.json "claim:difference.driving_score") points.\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert result.failures == []
    assert result.machine_checked == 1
    assert result.prose_audited == 0


def test_drifted_number_fails_naming_the_claim(tmp_path):
    write_report(tmp_path / "report.json", {"difference": {"driving_score": 16.34}})
    doc = tmp_path / "doc.md"
    doc.write_text(
        f'{CHECKED_MARKER}\n[17.01](report.json "claim:difference.driving_score")\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert len(result.failures) == 1
    assert "17.01" in result.failures[0]
    assert "difference.driving_score" in result.failures[0]
    assert "16.34" in result.failures[0]


def test_missing_artifact_fails_naming_the_path(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        f'{CHECKED_MARKER}\n[16.34](gone/report.json "claim:difference.driving_score")\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert len(result.failures) == 1
    assert "gone/report.json" in result.failures[0]


def test_bad_field_path_fails_naming_the_path(tmp_path):
    write_report(tmp_path / "report.json", {"difference": {"driving_score": 16.34}})
    doc = tmp_path / "doc.md"
    doc.write_text(
        f'{CHECKED_MARKER}\n[16.34](report.json "claim:difference.typo")\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert len(result.failures) == 1
    assert "difference.typo" in result.failures[0]


def test_prose_audited_claim_is_counted_but_not_number_checked(tmp_path):
    # The evidence is a Terraform file, not JSON — the checker must only
    # assert it exists, never try to parse a number out of it.
    (tmp_path / "kinesis.tf").write_text('resource "aws_kinesis_stream" "t" {}\n', encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text(
        f'{CHECKED_MARKER}\n[never applied to real AWS](kinesis.tf "claim:prose")\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert result.failures == []
    assert result.machine_checked == 0
    assert result.prose_audited == 1


def test_prose_audited_claim_still_requires_its_evidence_to_exist(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text(
        f'{CHECKED_MARKER}\n[never applied](gone.tf "claim:prose")\n', encoding="utf-8"
    )

    result = check_document(doc)

    assert len(result.failures) == 1
    assert "gone.tf" in result.failures[0]


def test_quoted_number_matches_at_its_own_precision(tmp_path):
    # The rule the write-up generators use: format the artifact value to the
    # quote's own decimal count. "0.40" is an honest 2-decimal quote of 0.3961.
    write_report(
        tmp_path / "report.json",
        {"score": 0.3961, "episodes": 10, "latency": 8.35},
    )
    doc = tmp_path / "doc.md"
    doc.write_text(
        f"{CHECKED_MARKER}\n"
        '[0.40](report.json "claim:score") over '
        '[10](report.json "claim:episodes") episodes at '
        '[8.35](report.json "claim:latency") ms, but not '
        '[0.41](report.json "claim:score").\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert result.machine_checked == 4
    assert len(result.failures) == 1
    assert "0.41" in result.failures[0]


def test_field_path_indexes_into_lists(tmp_path):
    # The real reports carry lists (e.g. the episode specs); a bare integer
    # segment indexes into one.
    write_report(tmp_path / "report.json", {"episodes": [{"seed": 1000}, {"seed": 1001}]})
    doc = tmp_path / "doc.md"
    doc.write_text(
        f"{CHECKED_MARKER}\n"
        'Seeds start at [1000](report.json "claim:episodes.0.seed") but there is no '
        '[1002](report.json "claim:episodes.2.seed").\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert len(result.failures) == 1
    assert "episodes.2.seed" in result.failures[0]


def test_fenced_code_blocks_are_not_claims(tmp_path):
    # docs/CLAIMS.md shows the syntax inside ``` fences; those examples must
    # be inert, or the convention page could never show a counter-example.
    write_report(tmp_path / "report.json", {"score": 1.5})
    doc = tmp_path / "doc.md"
    doc.write_text(
        f"{CHECKED_MARKER}\n"
        "```\n"
        '[9.9](report.json "claim:score")\n'
        '[16.34](gone.json claim:broken.example)\n'
        "```\n"
        '[1.5](report.json "claim:score")\n',
        encoding="utf-8",
    )

    result = check_document(doc)

    assert result.failures == []
    assert result.machine_checked == 1


def test_malformed_citation_fails_rather_than_silently_skipping(tmp_path):
    # A typo'd citation must not quietly become a non-claim with a green
    # build — that would defeat the checker's whole purpose.
    write_report(tmp_path / "report.json", {"difference": {"driving_score": 16.34}})
    doc = tmp_path / "doc.md"
    doc.write_text(
        f"{CHECKED_MARKER}\n[16.34](report.json claim:difference.driving_score)\n",
        encoding="utf-8",
    )

    result = check_document(doc)

    assert result.machine_checked == 0
    assert len(result.failures) == 1
    assert "malformed" in result.failures[0]


def test_boolean_field_never_matches_a_numeric_quote(tmp_path):
    write_report(tmp_path / "report.json", {"applied": True})
    doc = tmp_path / "doc.md"
    doc.write_text(
        f'{CHECKED_MARKER}\n[1](report.json "claim:applied")\n', encoding="utf-8"
    )

    result = check_document(doc)

    assert len(result.failures) == 1


def test_check_repo_only_checks_documents_that_opt_in(tmp_path):
    write_report(tmp_path / "report.json", {"score": 1.5})
    (tmp_path / "opted_in.md").write_text(
        f'{CHECKED_MARKER}\n[1.5](report.json "claim:score")\n', encoding="utf-8"
    )
    # Same drifted claim, but the document never opted in — ignored entirely.
    (tmp_path / "not_opted_in.md").write_text(
        '[9.9](report.json "claim:score")\n', encoding="utf-8"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "nested.md").write_text(
        f'{CHECKED_MARKER}\n[1.5](../report.json "claim:score")\n', encoding="utf-8"
    )

    result = check_repo(tmp_path)

    assert [report.document.name for report in result.documents] == ["opted_in.md", "nested.md"]
    assert result.failures == []
    assert result.machine_checked == 2


def test_cli_reports_coverage_counts_and_fails_on_drift(tmp_path, capsys):
    from pathfinder.claims import main

    write_report(tmp_path / "report.json", {"score": 1.5})
    (tmp_path / "doc.md").write_text(
        f"{CHECKED_MARKER}\n"
        '[1.5](report.json "claim:score") and [evidence](report.json "claim:prose")\n',
        encoding="utf-8",
    )

    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 machine-checked" in out
    assert "1 prose-audited" in out

    (tmp_path / "doc.md").write_text(
        f'{CHECKED_MARKER}\n[9.9](report.json "claim:score")\n', encoding="utf-8"
    )
    assert main([str(tmp_path)]) == 1
    assert "9.9" in capsys.readouterr().out


def test_claims_table_exists_and_carries_both_claim_kinds():
    """The claims table (issue #24) is the claim-of-record at the repo root.

    It must opt in to the checker and enumerate both kinds of claim: the
    numeric findings (machine-checked) and the negative/boundary claims
    (prose-audited). Exact counts would be brittle; what this pins is that
    the table exists, is checked, and neither kind has been emptied out.
    """
    table = REPO_ROOT / "CLAIMS.md"

    assert table.is_file()
    assert CHECKED_MARKER in table.read_text(encoding="utf-8")

    report = check_document(table)

    assert report.failures == []
    assert report.machine_checked > 0
    assert report.prose_audited > 0


def test_real_repo_documents_pass_the_checker():
    """The real gate: every opted-in document in this repo checks out.

    While no real document opts in this passes trivially, which is the
    behaviour issue #19 asks for; docs/CLAIMS.md's worked example opts in,
    so the convention page itself is held to its own rule.
    """
    result = check_repo(REPO_ROOT)

    assert result.failures == []
