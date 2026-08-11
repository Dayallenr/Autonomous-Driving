#!/usr/bin/env bash
# Bring up a local 3-node kind cluster and deploy the full PathFinder
# distributed-benchmark stack onto it: LocalStack (shared queue), the
# coordinator, an enqueue Job, and the worker Deployment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CLUSTER_NAME="pathfinder"
NAMESPACE="pathfinder"

if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  kind create cluster --name "$CLUSTER_NAME" --config k8s/kind-config.yaml
else
  echo "kind cluster '$CLUSTER_NAME' already exists, reusing it"
fi

echo "building images..."
docker build -t pathfinder-coordinator:latest -f Dockerfile.coordinator .
docker build -t pathfinder-worker:latest -f Dockerfile.worker .

echo "loading images into kind (they are not on any registry)..."
kind load docker-image pathfinder-coordinator:latest --name "$CLUSTER_NAME"
kind load docker-image pathfinder-worker:latest --name "$CLUSTER_NAME"

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

kubectl apply -f k8s/localstack.yaml
kubectl -n "$NAMESPACE" rollout status deployment/localstack --timeout=90s

kubectl apply -f k8s/coordinator.yaml
kubectl -n "$NAMESPACE" rollout status deployment/coordinator --timeout=60s

# Re-running this script re-applies the Job spec unchanged, which errors —
# delete any previous run's Job first so seeding is idempotent.
kubectl -n "$NAMESPACE" delete job enqueue-episodes --ignore-not-found
kubectl apply -f k8s/enqueue-job.yaml
kubectl -n "$NAMESPACE" wait --for=condition=complete --timeout=60s job/enqueue-episodes

kubectl apply -f k8s/worker.yaml
kubectl -n "$NAMESPACE" rollout status deployment/worker --timeout=60s

cat <<'EOF'

Cluster is up. Useful next steps:
  kubectl -n pathfinder get pods -o wide
  kubectl -n pathfinder logs -l app=worker --tail=100
  kubectl -n pathfinder port-forward svc/coordinator 50051:50051   # then query it from the host
  kubectl -n pathfinder scale deployment/worker --replicas=8       # elastic scale-out
  k8s/kind-down.sh                                                 # tear down when done
EOF
