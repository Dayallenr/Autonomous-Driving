# Customer-managed key for EKS secrets envelope encryption AND the cluster's
# CloudWatch log group. Costs $1/month while it exists (KMS CMKs are not
# free-tier) — small next to the cluster itself, and it's the difference
# between that data being encrypted with a key this account controls versus
# AWS's default (also encrypted, but not something you can audit/rotate/
# revoke independently).
resource "aws_kms_key" "eks_secrets" {
  description             = "PathFinder EKS Kubernetes secrets + log group encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.eks_secrets_kms.json
}

resource "aws_kms_alias" "eks_secrets" {
  name          = "alias/${var.cluster_name}-eks-secrets"
  target_key_id = aws_kms_key.eks_secrets.key_id
}

# Explicit rather than relying on the default key policy: this is exactly
# the two things that need to touch this key, and nothing else. CloudWatch
# Logs specifically requires this grant — a customer-managed key on a log
# group silently fails to encrypt without it.
data "aws_iam_policy_document" "eks_secrets_kms" {
  # "*" here is the standard AWS pattern for a KMS key *resource* policy —
  # unlike in an IAM identity policy, this key's resource policy is
  # inherently scoped to itself; "*" means "this key," not "every KMS key in
  # the account." kms:* for the account root is likewise the documented AWS
  # default (root always needs full administrative control of a key it
  # owns, or the key can become unmanageable if the policy is ever wrong).
  # checkov:skip=CKV_AWS_111:standard KMS resource-policy shape — "*" scopes to this key, not to every key in the account
  # checkov:skip=CKV_AWS_356:same — see comment above
  # checkov:skip=CKV_AWS_109:account root needs full control of its own key by AWS's own recommended pattern
  statement {
    sid       = "AccountRootFullAccess"
    effect    = "Allow"
    actions   = ["kms:*"]
    resources = ["*"]
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
  }

  statement {
    sid    = "CloudWatchLogsEncryption"
    effect = "Allow"
    actions = [
      "kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*",
      "kms:GenerateDataKey*", "kms:DescribeKey",
    ]
    resources = ["*"]
    principals {
      type        = "Service"
      identifiers = ["logs.${var.aws_region}.amazonaws.com"]
    }
    condition {
      test     = "ArnEquals"
      variable = "kms:EncryptionContext:aws:logs:arn"
      values   = ["arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/eks/${var.cluster_name}/cluster"]
    }
  }
}
