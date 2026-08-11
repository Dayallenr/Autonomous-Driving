#!/usr/bin/env bash
# Deploy PathFinder onto the real EKS cluster terraform/ provisions.
#
# Prerequisites (all manual, all deliberate — see CLAUDE.md's zero-spend
# rule): `terraform apply` has already run successfully in terraform/, and
# `aws configure` has already run with credentials that can reach this
# account.
#
# This script only touches Kubernetes objects on the cluster Terraform
# created — it does not run `terraform apply`/`destroy` itself.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

TF_DIR="terraform"
NAMESPACE="pathfinder"

tf_output() { terraform -chdir="$TF_DIR" output -raw "$1"; }

echo "reading terraform outputs..."
export AWS_REGION=$(tf_output aws_region)
CLUSTER_NAME=$(tf_output cluster_name)
export ECR_COORDINATOR_IMAGE="$(tf_output ecr_coordinator_repository_url):latest"
export ECR_WORKER_IMAGE="$(tf_output ecr_worker_repository_url):latest"
export SQS_QUEUE_URL=$(tf_output sqs_queue_url)
export KINESIS_STREAM_NAME=$(tf_output kinesis_stream_name)
export S3_TELEMETRY_BUCKET=$(tf_output s3_telemetry_bucket)
export WORKER_IRSA_ROLE_ARN=$(tf_output worker_irsa_role_arn)

echo "cluster: $CLUSTER_NAME  region: $AWS_REGION"
echo "pointing kubectl at the cluster..."
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$AWS_REGION"

echo "checking whether images have been pushed to ECR yet..."
if ! aws ecr describe-images --repository-name "${CLUSTER_NAME}-coordinator" --region "$AWS_REGION" >/dev/null 2>&1; then
  ECR_REGISTRY=$(tf_output ecr_coordinator_repository_url | cut -d/ -f1)
  cat <<PUSH
No images in ${CLUSTER_NAME}-coordinator yet. Push them first:
  aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
  docker build -t $ECR_COORDINATOR_IMAGE -f Dockerfile.coordinator .
  docker build -t $ECR_WORKER_IMAGE -f Dockerfile.worker .
  docker push $ECR_COORDINATOR_IMAGE
  docker push $ECR_WORKER_IMAGE
Then re-run this script.
PUSH
  exit 1
fi

render() { envsubst <"$1"; }

kubectl apply -f k8s/namespace.yaml
render k8s/eks/configmap.yaml | kubectl apply -f -
render k8s/eks/serviceaccount.yaml | kubectl apply -f -

render k8s/eks/coordinator.yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" rollout status deployment/coordinator --timeout=120s

kubectl -n "$NAMESPACE" delete job enqueue-episodes --ignore-not-found
render k8s/eks/enqueue-job.yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" wait --for=condition=complete --timeout=90s job/enqueue-episodes

render k8s/eks/worker.yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" rollout status deployment/worker --timeout=120s

cat <<'EOF'

Deployed. Useful next steps:
  kubectl -n pathfinder get pods -o wide
  kubectl -n pathfinder logs -l app=worker --tail=100 -f

  # watch progress (separate terminal):
  kubectl -n pathfinder port-forward svc/coordinator 50051:50051 &
  .venv/bin/python -c "
  import asyncio
  from pathfinder.rpc.client import CoordinatorClient
  async def main():
      client = CoordinatorClient.connect('localhost:50051')
      print(await client.get_run_status(''))
      await client.close()
  asyncio.run(main())
  "

  # once episodes_completed == episodes_total in that status:
  kubectl -n pathfinder scale deployment/worker --replicas=0   # stop workers before archiving
  k8s/eks/archive.sh                                            # drain telemetry to real S3 Parquet
  k8s/eks/evidence.sh                                           # capture results/aws_deployment/ artifacts
  cd terraform && terraform destroy                             # tear everything down when done
EOF
