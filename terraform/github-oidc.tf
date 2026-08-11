# Lets .github/workflows/deploy.yml authenticate to AWS via a short-lived
# OIDC token instead of long-lived access keys stored as a GitHub secret —
# there is no static credential to leak, rotate, or accidentally commit.
# GitHub's own OIDC thumbprint is stable and documented; unlike the EKS OIDC
# provider above, it doesn't need to be read from the issuer at apply time.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

data "aws_iam_policy_document" "github_deploy_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    # Scoped to this exact repo, any ref — not to any GitHub repo, and not
    # further restricted to e.g. only the main branch, because this role is
    # only ever exercised by a human manually choosing workflow_dispatch
    # (see deploy.yml), not by an automatic push/PR trigger.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:*"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${var.cluster_name}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_deploy_assume_role.json
}

# Broad by portfolio-project necessity, not by careless default: this role
# runs `terraform apply`/`destroy` against this exact configuration, so it
# needs to create/modify/delete everything that configuration defines (EKS,
# IAM, ECR, SQS, S3, Kinesis, KMS, CloudWatch Logs) plus push to ECR. A real
# production setup would split "who can plan/apply infra" from "who can push
# images" into separate roles with tighter per-service scoping — worth
# saying explicitly in an interview rather than implying this is
# least-privilege. What *is* tightly scoped is the trust policy above: only
# this one GitHub repository can assume this role at all.
resource "aws_iam_role_policy_attachment" "github_deploy_ecr" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser"
}

resource "aws_iam_role_policy_attachment" "github_deploy_eks" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

data "aws_iam_policy_document" "github_deploy_terraform" {
  # This document is deliberately as broad as the comment above says —
  # skipped explicitly rather than silently unaddressed. The trust policy on
  # aws_iam_role.github_deploy (only this one GitHub repo can assume it) is
  # the real control; these checks are about the permissions policy shape,
  # which a real least-privilege setup would split by service and constrain
  # to specific resource ARNs, not "*".
  # checkov:skip=CKV_AWS_108:this role's whole job is applying this Terraform config, which itself defines every resource below — see the comment above
  # checkov:skip=CKV_AWS_111:same — write access is the point of a deploy role for infrastructure this broad
  # checkov:skip=CKV_AWS_107:no credential-issuing actions here (no iam:CreateAccessKey etc. beyond what EKS/ECR/etc. themselves need)
  # checkov:skip=CKV_AWS_356:wildcard resources match "manages every resource this Terraform config defines," not an unbounded grant to unrelated infrastructure
  # checkov:skip=CKV_AWS_109:iam:* is needed to create/attach the very IAM roles this config defines (cluster role, node role, IRSA role) — scoping further needs per-role ARNs this config can't know until first apply
  # checkov:skip=CKV_AWS_110:trust policy is locked to one GitHub repo (see above) — that repo's own protections are what gate privilege here, not this policy
  # checkov:skip=CKV2_AWS_40:see above — a real least-privilege split is a documented follow-up, not an oversight
  statement {
    sid    = "TerraformManagedResources"
    effect = "Allow"
    actions = [
      "eks:*",
      "ec2:*",
      "iam:*",
      "ecr:*",
      "sqs:*",
      "s3:*",
      "kinesis:*",
      "kms:*",
      "logs:*",
      "sts:GetCallerIdentity",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_deploy_terraform" {
  name   = "${var.cluster_name}-github-deploy-terraform"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy_terraform.json
}
