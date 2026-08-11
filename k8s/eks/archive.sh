#!/usr/bin/env bash
# Run after workers finish (see deploy.sh's printed next steps): drains the
# real Kinesis stream into real S3 Parquet.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TF_DIR="terraform"
NAMESPACE="pathfinder"
tf_output() { terraform -chdir="$TF_DIR" output -raw "$1"; }

export ECR_WORKER_IMAGE="$(tf_output ecr_worker_repository_url):latest"

kubectl -n "$NAMESPACE" delete job archive-telemetry --ignore-not-found
envsubst <k8s/eks/archive-job.yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" wait --for=condition=complete --timeout=120s job/archive-telemetry
kubectl -n "$NAMESPACE" logs job/archive-telemetry
