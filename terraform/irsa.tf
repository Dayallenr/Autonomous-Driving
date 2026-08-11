# IRSA: worker Pods assume this role via their Kubernetes ServiceAccount
# (annotated with its ARN — see k8s/eks/serviceaccount.yaml) instead of
# carrying static AWS access keys baked into the container or a Secret. The
# trust policy below is scoped to exactly one namespace/ServiceAccount name,
# and the permissions policy is scoped to exactly the three resources this
# project's workers actually touch — no wildcard resource ARNs.

data "aws_iam_policy_document" "worker_irsa_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.eks.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:sub"
      values   = ["system:serviceaccount:pathfinder:pathfinder-worker"]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(aws_iam_openid_connect_provider.eks.url, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "worker_irsa" {
  name               = "${var.cluster_name}-worker-irsa"
  assume_role_policy = data.aws_iam_policy_document.worker_irsa_assume_role.json
}

data "aws_iam_policy_document" "worker_permissions" {
  statement {
    sid    = "WorkQueue"
    effect = "Allow"
    actions = [
      "sqs:SendMessage",
      "sqs:ReceiveMessage",
      "sqs:DeleteMessage",
      "sqs:GetQueueAttributes",
      "sqs:CreateQueue", # scripts/run_worker.py's ensure_queue() convenience path; a no-op once the queue already exists
    ]
    resources = [aws_sqs_queue.episodes.arn, aws_sqs_queue.dead_letter.arn]
  }

  statement {
    sid    = "Telemetry"
    effect = "Allow"
    actions = [
      "kinesis:PutRecord",
      "kinesis:PutRecords",
      "kinesis:DescribeStreamSummary",
      "kinesis:ListShards",
      "kinesis:GetShardIterator",
      "kinesis:GetRecords",
    ]
    resources = [aws_kinesis_stream.telemetry.arn]
  }

  statement {
    sid    = "Warehouse"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [aws_s3_bucket.telemetry.arn, "${aws_s3_bucket.telemetry.arn}/*"]
  }

  statement {
    # Kinesis's own key, not the S3/SQS server-side-encryption defaults —
    # PutRecord/GetRecords need Decrypt/GenerateDataKey against whatever key
    # the stream is encrypted with.
    sid       = "KinesisKms"
    effect    = "Allow"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = ["arn:aws:kms:${var.aws_region}:${data.aws_caller_identity.current.account_id}:alias/aws/kinesis"]
  }
}

resource "aws_iam_role_policy" "worker_permissions" {
  name   = "${var.cluster_name}-worker-permissions"
  role   = aws_iam_role.worker_irsa.id
  policy = data.aws_iam_policy_document.worker_permissions.json
}
