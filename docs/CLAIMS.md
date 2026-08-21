# Citing claims: the convention the checker enforces

<!-- claims: checked -->

Every load-bearing claim in this repo's documents must correspond to a real
artifact a reader can verify — that is the prime directive. This page defines
the citation convention that makes the correspondence machine-checkable, and
`pathfinder/claims.py` is the checker that enforces it. The checker runs as an
ordinary test (`tests/test_claims_checker.py`), so a document that drifts from
its artifact is a failing build, not a stale sentence.

This page opts in to the checker, so the worked examples below are themselves
verified in CI.

## Opting a document in

Put this marker anywhere in the document (conventionally near the top):

```
<!-- claims: checked -->
```

The checker scans the repo's root-level `*.md` files and everything under
`docs/`; only documents carrying the marker are checked. Generated write-ups
under `results/` are artifacts, not documents, and stay out of scope.
Citations inside fenced code blocks are inert — that is how this page shows
the syntax without the examples being live — and a link that mentions
`claim:` without matching the syntax exactly is a checker failure, never a
silent skip.

## Citing a number (machine-checked)

A numeric claim is an ordinary Markdown link:

```
[16.34](../results/ablation/carla_report.json "claim:difference.driving_score")
```

- **Link text** — the quoted number, exactly as displayed, nothing else.
  Units and prose stay outside the link, and thousands separators are not
  supported — write `5978`, never `5,978`.
- **Link target** — the artifact's path, relative to the containing document
  (like any Markdown link), so a citation always renders as a working link.
- **Link title** — `claim:` followed by the dotted field path into the
  artifact's JSON. A bare integer segment indexes into a list
  (`episodes.0.seed`).

The checker asserts the artifact exists, the field path resolves, and the
quoted number equals the field's value **rounded to the quote's own decimal
count** — Python's `format` rounding, the same the write-up generators use. So
[16.34](../results/ablation/carla_report.json "claim:difference.driving_score")
must match the field exactly at two decimals, while a coarser quote of the same
run's baseline score, say
[25](../results/ablation/carla_report.json "claim:baseline.summary.driving_score"),
is honest at zero decimals.

## Citing a non-numeric claim (prose-audited)

Negative and boundary claims — "never applied to real AWS", "not project
work", "structurally unmeasurable" — have no number to check. Cite them with
the field path `prose`:

```
[never applied to real AWS](../terraform/kinesis.tf "claim:prose")
```

The checker verifies the evidence path exists and counts the claim as
prose-audited; the sentence itself is held true by review, not by machine. A
prose citation should point at the most load-bearing evidence for the claim —
the Terraform that was never applied, the report that records a skip, the test
that pins a boundary. Worked example:
[Kinesis is provisioned in Terraform but never applied](../terraform/kinesis.tf "claim:prose").

## Running the checker

```
.venv/bin/python -m pytest tests/test_claims_checker.py   # the CI gate
.venv/bin/python -m pathfinder.claims                     # coverage counts
```

The CLI prints machine-checked vs prose-audited counts per document and in
total, so coverage — how much of a document is held to evidence — is visible
at a glance.
