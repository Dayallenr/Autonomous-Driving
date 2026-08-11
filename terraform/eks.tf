# Hand-written EKS resources rather than a community module (e.g.
# terraform-aws-modules/eks) — more lines, but every resource here is one I
# can point at and explain, which matters more for this project than saving
# fifty lines behind an abstraction I'd be quoting from memory in an
# interview.

data "aws_caller_identity" "current" {}

# ─────────────────────────────────────────────────────────────────────────────
# Cluster IAM role
# ─────────────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "eks_cluster_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_cluster" {
  name               = "${var.cluster_name}-eks-cluster"
  assume_role_policy = data.aws_iam_policy_document.eks_cluster_assume_role.json
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# ─────────────────────────────────────────────────────────────────────────────
# Cluster
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "eks_cluster" {
  name       = "/aws/eks/${var.cluster_name}/cluster"
  kms_key_id = aws_kms_key.eks_secrets.arn
  # 7 days, not the 1-year checkov recommends for production audit logging:
  # this is a demo cluster destroyed the same session, so nothing outlives
  # the log group anyway. #checkov:skip=CKV_AWS_338:demo cluster, torn down same session — retention length is moot
  retention_in_days = 7
}

resource "aws_eks_cluster" "this" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  # A fully private endpoint (what checkov's CKV_AWS_39 wants) needs a
  # bastion or VPN to reach at all — real infrastructure this project
  # deliberately doesn't stand up for a cluster that lives for one session.
  # public_access_cidrs (CKV_AWS_38) is the actual lever here: narrow it in
  # terraform.tfvars instead of leaving the default wide open.
  # #checkov:skip=CKV_AWS_39:no bastion/VPN in this project — public endpoint is how kubectl reaches the cluster at all
  # #checkov:skip=CKV_AWS_38:defaults wide open since Terraform can't know your IP — set cluster_endpoint_public_access_cidrs in terraform.tfvars to narrow it before applying
  vpc_config {
    subnet_ids              = data.aws_subnets.default.ids
    endpoint_public_access  = true
    endpoint_private_access = false
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
  }

  encryption_config {
    resources = ["secrets"]
    provider {
      key_arn = aws_kms_key.eks_secrets.arn
    }
  }

  # All five log types, sent to the log group above — the control-plane
  # observability a real cluster needs, at the cost of a few cents of
  # CloudWatch Logs ingestion for a short demo run.
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_cloudwatch_log_group.eks_cluster,
  ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Node group IAM role
# ─────────────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "eks_node_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eks_node" {
  name               = "${var.cluster_name}-eks-node"
  assume_role_policy = data.aws_iam_policy_document.eks_node_assume_role.json
}

resource "aws_iam_role_policy_attachment" "eks_node_worker_policy" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_node_cni_policy" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "eks_node_ecr_readonly" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# ─────────────────────────────────────────────────────────────────────────────
# Node group
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_eks_node_group" "default" {
  cluster_name    = aws_eks_cluster.this.name
  node_group_name = "default"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = data.aws_subnets.default.ids
  instance_types  = [var.node_instance_type]
  # On-demand, not spot: spot is cheaper but can be reclaimed mid-demo, and
  # this cluster only lives for one short session anyway — the reliability
  # is worth more here than the marginal saving.
  capacity_type = "ON_DEMAND"

  scaling_config {
    desired_size = var.node_desired_size
    min_size     = var.node_min_size
    max_size     = var.node_max_size
  }

  update_config {
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_node_worker_policy,
    aws_iam_role_policy_attachment.eks_node_cni_policy,
    aws_iam_role_policy_attachment.eks_node_ecr_readonly,
  ]
}

# ─────────────────────────────────────────────────────────────────────────────
# IRSA (IAM Roles for Service Accounts) — the OIDC provider Pods need to
# assume scoped IAM roles instead of carrying static AWS keys.
# ─────────────────────────────────────────────────────────────────────────────

data "tls_certificate" "eks_oidc" {
  url = aws_eks_cluster.this.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  url             = aws_eks_cluster.this.identity[0].oidc[0].issuer
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks_oidc.certificates[0].sha1_fingerprint]
}
