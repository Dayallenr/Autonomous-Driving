provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "pathfinder"
      ManagedBy = "terraform"
    }
  }
}

# The EKS/IRSA OIDC trust policies need the cluster's own TLS certificate
# thumbprint, which the aws provider doesn't expose — the tls provider reads
# it directly from the OIDC issuer once the cluster exists.
provider "tls" {}
