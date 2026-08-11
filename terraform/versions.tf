terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }

  # Local state deliberately, not S3+DynamoDB: a remote backend needs its own
  # bucket/table provisioned first (a chicken-and-egg problem), and this
  # project is applied by one person in one sitting, not a team needing
  # shared/locked state. Note the tradeoff, don't hide it: `terraform.tfstate`
  # is gitignored, so losing this machine loses the state — acceptable for a
  # short-lived demo cluster that gets destroyed the same session, not
  # acceptable for anything meant to persist.
}
