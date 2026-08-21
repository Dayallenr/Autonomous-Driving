"""The claim checker (issue #19): hold documents to their artifacts.

The citation convention is documented in ``docs/CLAIMS.md``. In short: a
document opts in with the marker ``<!-- claims: checked -->``, and every cited
claim is an ordinary Markdown link whose *title* carries the claim:

    [16.34](results/ablation/carla_report.json "claim:difference.driving_score")
    [never applied to real AWS](../terraform/kinesis.tf "claim:prose")

The first kind is machine-checked: the artifact must exist, the field path must
resolve, and the quoted number must equal the field's value rounded to the
quote's own decimal count — the same Python ``format`` rounding the write-up
generators use. The second kind is prose-audited: the checker verifies the
evidence path exists and counts the claim, but checks no number.

Link targets resolve relative to the containing document, exactly as Markdown
renders them, so a citation is always also a working link.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Claim", "DocumentReport", "RepoReport", "check_document", "check_repo"]

CHECKED_MARKER = "<!-- claims: checked -->"
PROSE_FIELD = "prose"

# An inline Markdown link with a quoted title: [text](target "title").
_LINK = re.compile(r'\[([^\]]+)\]\(\s*([^()\s"]+)\s+"claim:([^"]+)"\s*\)')


@dataclass(frozen=True)
class Claim:
    """One cited claim as parsed from a document."""

    document: Path
    line: int
    quoted: str
    artifact: str
    field_path: str  # PROSE_FIELD marks a prose-audited claim


@dataclass
class DocumentReport:
    """What the checker found in one opted-in document."""

    document: Path
    claims: list[Claim] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def machine_checked(self) -> int:
        return sum(1 for claim in self.claims if claim.field_path != PROSE_FIELD)

    @property
    def prose_audited(self) -> int:
        return sum(1 for claim in self.claims if claim.field_path == PROSE_FIELD)


@dataclass
class RepoReport:
    """Every opted-in document's result, plus the coverage counts."""

    documents: list[DocumentReport] = field(default_factory=list)

    @property
    def failures(self) -> list[str]:
        return [failure for report in self.documents for failure in report.failures]

    @property
    def machine_checked(self) -> int:
        return sum(report.machine_checked for report in self.documents)

    @property
    def prose_audited(self) -> int:
        return sum(report.prose_audited for report in self.documents)


def check_repo(root: Path) -> RepoReport:
    """Check every opted-in Markdown document under ``root``.

    A document opts in by containing ``CHECKED_MARKER``. Candidates are the
    root-level ``*.md`` files (README, the future claims table) and everything
    under ``docs/`` — generated write-ups in ``results/`` cite nothing and are
    deliberately out of scope.
    """
    candidates = sorted(root.glob("*.md")) + sorted((root / "docs").rglob("*.md"))
    return RepoReport(
        documents=[
            check_document(document)
            for document in candidates
            if CHECKED_MARKER in document.read_text(encoding="utf-8")
        ]
    )


def find_claims(document: Path) -> list[Claim]:
    """Parse every cited claim out of one document, in order."""
    text = document.read_text(encoding="utf-8")
    return [
        Claim(
            document=document,
            line=number,
            quoted=match.group(1),
            artifact=match.group(2),
            field_path=match.group(3),
        )
        for number, line in enumerate(text.splitlines(), start=1)
        for match in _LINK.finditer(line)
    ]


def check_document(document: Path) -> DocumentReport:
    """Check every cited claim in one document against its artifacts."""
    report = DocumentReport(document=document)
    report.claims = find_claims(document)
    for claim in report.claims:
        failure = _check_claim(claim)
        if failure is not None:
            report.failures.append(failure)
    return report


def _check_claim(claim: Claim) -> str | None:
    artifact = (claim.document.parent / claim.artifact).resolve()
    if not artifact.is_file():
        return (
            f"{claim.document.name}:{claim.line}: claim [{claim.quoted}] cites "
            f"missing artifact {claim.artifact}"
        )
    if claim.field_path == PROSE_FIELD:
        return None
    import json

    data = json.loads(artifact.read_text(encoding="utf-8"))
    try:
        value = _resolve(data, claim.field_path)
    except LookupError:
        return (
            f"{claim.document.name}:{claim.line}: claim [{claim.quoted}] cites "
            f"field path {claim.field_path}, which does not exist in {claim.artifact}"
        )
    if not _quoted_matches(claim.quoted, value):
        return (
            f"{claim.document.name}:{claim.line}: claim [{claim.quoted}] does not "
            f"match {claim.artifact}:{claim.field_path} = {value!r}"
        )
    return None


def _resolve(data: object, field_path: str) -> object:
    value = data
    for segment in field_path.split("."):
        if isinstance(value, list) and segment.isdigit() and int(segment) < len(value):
            value = value[int(segment)]
        elif isinstance(value, dict) and segment in value:
            value = value[segment]
        else:
            raise KeyError(segment)
    return value


def _quoted_matches(quoted: str, value: object) -> bool:
    decimals = len(quoted.partition(".")[2])
    return isinstance(value, (int, float)) and f"{value:.{decimals}f}" == quoted


def main(argv: list[str] | None = None) -> int:
    """Check every opted-in document and print the coverage counts."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check every opted-in document's cited claims against their artifacts."
    )
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path.cwd(), help="repo root (default: cwd)"
    )
    args = parser.parse_args(argv)

    result = check_repo(args.root)
    for report in result.documents:
        print(
            f"{report.document.relative_to(args.root)}: {report.machine_checked} "
            f"machine-checked, {report.prose_audited} prose-audited"
        )
    for failure in result.failures:
        print(f"FAIL {failure}")
    print(
        f"total: {result.machine_checked} machine-checked, {result.prose_audited} "
        f"prose-audited, {len(result.failures)} failure(s)"
    )
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
