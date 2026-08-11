#!/usr/bin/env bash
# Delete the local kind cluster entirely (not just the workloads on it) — it's
# local/free either way, but leaving it around wastes Docker Desktop resources.
set -euo pipefail

CLUSTER_NAME="pathfinder"
kind delete cluster --name "$CLUSTER_NAME"
