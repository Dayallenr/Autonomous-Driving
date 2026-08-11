# Reuse the account's default VPC rather than provisioning a new one. A
# custom VPC would need a NAT Gateway for private-subnet nodes to reach the
# internet (pull images, talk to the EKS API) — that's ~$0.045/hr plus
# per-GB data processing, billing for as long as it exists, independent of
# whether anything is using it. The default VPC's subnets are public with an
# Internet Gateway already attached, which is fine for a cluster that's
# applied, demoed, and destroyed in one sitting.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
