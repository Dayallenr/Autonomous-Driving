#!/usr/bin/env bash
# Capture proof this actually ran on real AWS, before `terraform destroy`
# removes the evidence along with the infrastructure. This is what backs the
# resume/README claim — not the claim itself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TF_DIR="terraform"
NAMESPACE="pathfinder"
tf_output() { terraform -chdir="$TF_DIR" output -raw "$1"; }

OUT_DIR="results/aws_deployment/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUT_DIR"

kubectl -n "$NAMESPACE" get pods -o wide >"$OUT_DIR/pods.txt"
kubectl -n "$NAMESPACE" get deployment,job,svc -o wide >"$OUT_DIR/resources.txt"
kubectl -n "$NAMESPACE" logs -l app=worker --tail=1000 >"$OUT_DIR/worker_logs.txt" || true
kubectl -n "$NAMESPACE" logs job/archive-telemetry >"$OUT_DIR/archive_log.txt" || true

S3_BUCKET=$(tf_output s3_telemetry_bucket)
AWS_REGION=$(tf_output aws_region)
aws s3 ls "s3://$S3_BUCKET/telemetry/" --recursive --region "$AWS_REGION" >"$OUT_DIR/s3_listing.txt" || true

echo "evidence written to $OUT_DIR:"
ls -la "$OUT_DIR"
