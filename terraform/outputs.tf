output "aws_region" {
  value = var.aws_region
}

output "cluster_name" {
  value = aws_eks_cluster.this.name
}

output "cluster_endpoint" {
  value = aws_eks_cluster.this.endpoint
}

output "configure_kubectl" {
  description = "Command to point kubectl at this cluster."
  value       = "aws eks update-kubeconfig --name ${aws_eks_cluster.this.name} --region ${var.aws_region}"
}

output "ecr_coordinator_repository_url" {
  value = aws_ecr_repository.coordinator.repository_url
}

output "ecr_worker_repository_url" {
  value = aws_ecr_repository.worker.repository_url
}

output "sqs_queue_url" {
  value = aws_sqs_queue.episodes.id
}

output "sqs_dead_letter_queue_url" {
  value = aws_sqs_queue.dead_letter.id
}

output "s3_telemetry_bucket" {
  value = aws_s3_bucket.telemetry.bucket
}

output "kinesis_stream_name" {
  value = aws_kinesis_stream.telemetry.name
}

output "worker_irsa_role_arn" {
  description = "Annotate the pathfinder-worker ServiceAccount with this (see k8s/eks/serviceaccount.yaml)."
  value       = aws_iam_role.worker_irsa.arn
}

output "github_deploy_role_arn" {
  description = "Set as the AWS_DEPLOY_ROLE_ARN repo variable for .github/workflows/deploy.yml."
  value       = aws_iam_role.github_deploy.arn
}
