#!/usr/bin/env python
"""
Regenerate gRPC stubs from protos/pathfinder.proto.

    python scripts/generate_protos.py

Why this wrapper instead of calling protoc directly: ``grpc_tools.protoc``
emits ``import sentinel_pb2`` at the top of ``sentinel_pb2_grpc.py`` — a flat
import that only resolves if the output directory happens to be on sys.path.
Inside a package it raises ``ModuleNotFoundError``. This is a long-standing
protoc behaviour, and every project that vendors generated stubs has to rewrite
that line. Doing it in a checked-in script keeps the fix reproducible instead of
being a manual edit someone re-does after every regeneration.

Generated files are committed so that neither the runtime nor CI needs
``grpc_tools`` installed to import the package.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = REPO_ROOT / "protos"
OUT_DIR = REPO_ROOT / "pathfinder" / "rpc" / "generated"

PACKAGE_HEADER = '''"""Generated gRPC stubs — do not edit by hand.

Regenerate with: python scripts/generate_protos.py
"""
'''


def main() -> int:
    protos = sorted(PROTO_DIR.glob("*.proto"))
    if not protos:
        print(f"no .proto files in {PROTO_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--grpc_python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        *[str(path) for path in protos],
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        print(
            "\nprotoc failed. Install the toolchain with: pip install grpcio-tools",
            file=sys.stderr,
        )
        return result.returncode

    # Rewrite flat imports to explicit relative imports.
    rewritten = 0
    for path in OUT_DIR.glob("*_pb2_grpc.py"):
        source = path.read_text(encoding="utf-8")
        patched = re.sub(
            r"^import (\w+_pb2) as (\w+)$",
            r"from . import \1 as \2",
            source,
            flags=re.MULTILINE,
        )
        if patched != source:
            path.write_text(patched, encoding="utf-8")
            rewritten += 1

    (OUT_DIR / "__init__.py").write_text(PACKAGE_HEADER, encoding="utf-8")

    print(f"generated stubs for {len(protos)} proto file(s) in {OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"rewrote flat imports in {rewritten} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
