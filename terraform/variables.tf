variable "aws_region" {
  description = "AWS region for every resource in this configuration."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name."
  type        = string
  default     = "pathfinder"
}

variable "kubernetes_version" {
  description = "EKS control plane Kubernetes version."
  type        = string
  default     = "1.30"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "CIDRs allowed to reach the EKS API's public endpoint. Defaults wide open because this is a single-operator demo with no VPN/bastion and no way for Terraform to know your current IP — narrow this to your own IP (e.g. [\"203.0.113.4/32\"]) in terraform.tfvars before applying if you want the tighter posture checkov recommends."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "node_instance_type" {
  description = "EC2 instance type for the managed node group. t3.medium is the cheapest type with enough memory to run the coordinator + several worker pods comfortably."
  type        = string
  default     = "t3.medium"
}

variable "node_desired_size" {
  description = "Desired worker node count. 2 nodes is enough to demonstrate real cross-node pod scheduling without idling extra EC2 spend."
  type        = number
  default     = 2
}

variable "node_min_size" {
  type    = number
  default = 1
}

variable "node_max_size" {
  description = "Upper bound if this were scaled out further. Kept small deliberately — this cluster is meant to be applied, demoed, and destroyed the same session, not to autoscale unattended."
  type        = number
  default     = 4
}

variable "github_repository" {
  description = "GitHub repo allowed to assume the deploy role via OIDC, as \"owner/repo\". Required non-default so the trust policy is never accidentally left open to any repository."
  type        = string
}

variable "sqs_visibility_timeout_seconds" {
  type    = number
  default = 30
}

variable "sqs_max_receive_count" {
  description = "Deliveries before a message moves to the dead-letter queue. Matches pathfinder.cloud.queue's default so the real queue's redrive behavior matches what's already tested against the local backend."
  type        = number
  default     = 3
}

variable "kinesis_retention_hours" {
  description = "How long Kinesis retains records. 24h (the minimum) — this stream is provisioned for one short demo, not standing telemetry infrastructure."
  type        = number
  default     = 24
}
