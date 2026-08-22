# PathFinder — real AWS deployment runbook

Everything in this directory is validated (`terraform validate`, `terraform
fmt -check`, `checkov`) but has **never been applied**. Nothing here costs
money until `terraform apply` is run, and that should only happen in one
focused sitting, ending in `terraform destroy` the same session.

## What this provisions

EKS cluster (2× `t3.medium` nodes, on-demand), ECR repos for both images, a
real SQS queue + DLQ, a real S3 bucket (7-day auto-expiry), a real Kinesis
stream (on-demand mode), IRSA for worker pods (no static AWS keys in the
cluster), a GitHub OIDC deploy role for `.github/workflows/deploy.yml`, and
the ground-rules $1 zero-spend budget alarm (`budget.tf`).

## The one sanctioned subset: the SQS wizard (issue #18)

`scripts/sqs_apply_wizard.sh` (run it from anywhere; it finds this
directory) is the only supported way to apply *part* of this configuration:
it creates the budget alarm first — and refuses to continue until the
Budgets API confirms it — then targeted-applies exactly the episode queue,
its DLQ, and the DLQ's redrive-allow policy. That footprint costs $0/month
(SQS permanent free tier; a notification-only budget is free), and
`tests/test_sqs_wizard.py` pins the wizard's target set so it can never
grow to touch anything else. Everything below this line — the full cluster
— remains validated-never-applied until the sitting described next.

## Rough cost for one session (apply → demo → destroy within ~2 hours)

| Resource | Rate | ~2hr cost |
|---|---|---|
| EKS control plane | $0.10/hr | $0.20 |
| 2× t3.medium (on-demand) | $0.0416/hr each | $0.17 |
| KMS CMK | $1/month, prorated | ~$0.003 |
| Kinesis (on-demand) | $0.04/hr + per-request | ~$0.10 |
| S3, SQS, ECR, CloudWatch Logs | free tier / negligible | ~$0.01 |
| **Total** | | **≈ $0.50** |

Leaving it running longer costs more, roughly linearly (EKS + nodes ≈
$0.15/hr combined) — the risk isn't the rate, it's forgetting to destroy it.

## Sequence

```bash
# 1. Prep (you): AWS credentials configured, e.g.
aws configure

# 2. terraform.tfvars — at minimum set github_repository; consider
#    narrowing cluster_endpoint_public_access_cidrs to your own IP.
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars

# 3. Plan and review before applying anything
cd terraform
terraform init
terraform plan

# 4. Apply — this is the money step. Takes ~10-15 min (EKS control plane
#    provisioning is the slow part).
terraform apply

# 5. Push images and deploy the workload (from repo root)
cd ..
aws eks update-kubeconfig --name pathfinder --region us-east-1
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin "$(terraform -chdir=terraform output -raw ecr_coordinator_repository_url | cut -d/ -f1)"
docker build -t "$(terraform -chdir=terraform output -raw ecr_coordinator_repository_url):latest" -f Dockerfile.coordinator .
docker build -t "$(terraform -chdir=terraform output -raw ecr_worker_repository_url):latest" -f Dockerfile.worker .
docker push "$(terraform -chdir=terraform output -raw ecr_coordinator_repository_url):latest"
docker push "$(terraform -chdir=terraform output -raw ecr_worker_repository_url):latest"

k8s/eks/deploy.sh   # applies namespace/configmap/serviceaccount/coordinator/enqueue-job/worker

# 6. Watch it run (deploy.sh prints the exact commands), then once
#    episodes_completed == episodes_total:
kubectl -n pathfinder scale deployment/worker --replicas=0
k8s/eks/archive.sh    # drains Kinesis -> real S3 Parquet
k8s/eks/evidence.sh   # writes results/aws_deployment/<timestamp>/ — commit this

# 7. Tear down. Every one of these matters — see the checklist below.
cd terraform
terraform destroy
```

## Teardown verification checklist

Run through this **every time**, right after `terraform destroy` reports
success — Terraform tracks what it created, but a stray manual resource or a
partial destroy can leave something billing silently.

- [ ] `aws eks list-clusters --region us-east-1` — empty (or doesn't list `pathfinder`)
- [ ] `aws ec2 describe-instances --region us-east-1 --filters "Name=tag:Project,Values=pathfinder" "Name=instance-state-name,Values=running,pending"` — empty
- [ ] `aws ec2 describe-volumes --region us-east-1 --filters "Name=tag:Project,Values=pathfinder"` — empty (EBS volumes sometimes survive node termination)
- [ ] `aws ec2 describe-addresses --region us-east-1` — no unattached Elastic IPs
- [ ] `aws kinesis list-streams --region us-east-1` — `pathfinder-telemetry` not present
- [ ] `aws sqs list-queues --region us-east-1 --queue-name-prefix pathfinder` — empty
- [ ] `aws s3 ls | grep pathfinder` — bucket gone (or emptying + the 7-day lifecycle rule will finish the job if `destroy` couldn't empty it)
- [ ] AWS Cost Explorer / Billing dashboard — no unexpected new line item the next day

If `terraform destroy` fails partway (it can, if e.g. the S3 bucket wasn't
empty), re-run it — Terraform is idempotent about this — rather than deleting
resources manually out from under its state.

## What's deliberately NOT least-privilege

The `github_deploy` IAM role (`terraform/github-oidc.tf`) has broad
permissions across every service this config touches, because its whole job
is applying this config. The trust policy — only `github_repository` can
assume it at all — is the real boundary. Said explicitly here and in the
Terraform comments so it reads as a documented scoping decision, not an
oversight, in an interview.
