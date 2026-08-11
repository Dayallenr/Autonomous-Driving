# AES256 (ECR's default at-rest encryption), not a customer-managed KMS key,
# for both repos below: these images have no secrets baked in (see both
# Dockerfiles — just source + deps), so the audit/revocation control a CMK
# buys isn't worth a second KMS key's cost and the extra IAM wiring ECR-KMS
# access needs.

resource "aws_ecr_repository" "coordinator" {
  # checkov:skip=CKV_AWS_136:no secrets in these images; AES256 default encryption is sufficient here
  name                 = "${var.cluster_name}-coordinator"
  image_tag_mutability = "IMMUTABLE" # a tag is a specific build, not a moving pointer — re-pushing the same tag should fail loudly, not silently replace what's running

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "worker" {
  # checkov:skip=CKV_AWS_136:no secrets in these images; AES256 default encryption is sufficient here
  name                 = "${var.cluster_name}-worker"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Untagged images (superseded by a later push, or a failed CI run) are pure
# storage cost with no way to reference them — expire them automatically
# instead of relying on someone remembering to prune.
locals {
  ecr_lifecycle_policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "expire untagged images after 7 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 7
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "coordinator" {
  repository = aws_ecr_repository.coordinator.name
  policy     = local.ecr_lifecycle_policy
}

resource "aws_ecr_lifecycle_policy" "worker" {
  repository = aws_ecr_repository.worker.name
  policy     = local.ecr_lifecycle_policy
}
