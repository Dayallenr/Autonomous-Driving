# Object store for TelemetryWarehouse's Parquet output and DatasetRegistry
# versions (pathfinder/cloud/objects.py, warehouse.py). Bucket names are
# globally unique across all of AWS, so the account id is baked into the
# name rather than risking a collision with someone else's "pathfinder"
# bucket.
#
# This bucket auto-expires its objects after 7 days (below) and is meant to
# be destroyed with the rest of this stack the same session it's applied.
# Several checkov S3 hardening checks assume standing production
# infrastructure and don't fit that — documented and skipped explicitly
# below, inside the resource block, rather than silently unaddressed.
resource "aws_s3_bucket" "telemetry" {
  # checkov:skip=CKV_AWS_21:versioning is for protecting long-lived data from accidental overwrite; this bucket's objects expire in 7 days by design
  # checkov:skip=CKV_AWS_145:AES256 (below) is sufficient for short-lived demo output; a customer-managed KMS key adds cost/complexity for no real gain here
  # checkov:skip=CKV_AWS_18:access logging is for auditing standing infrastructure over time; nothing about a 7-day demo bucket needs an access trail
  # checkov:skip=CKV2_AWS_62:event notifications have no consumer in this project — there's no pipeline downstream of this bucket to notify
  # checkov:skip=CKV_AWS_144:cross-region replication is a production DR concern; this bucket's content is reproducible by re-running the benchmark
  bucket = "${var.cluster_name}-telemetry-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket_public_access_block" "telemetry" {
  bucket                  = aws_s3_bucket.telemetry.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "telemetry" {
  bucket = aws_s3_bucket.telemetry.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Auto-delete after a short window: this bucket exists for one demo session's
# worth of Parquet output, not standing warehouse infrastructure. Prevents
# it from quietly accumulating storage cost if a `terraform destroy` is ever
# skipped or the bucket is emptied but the account keeps writing to it later.
resource "aws_s3_bucket_lifecycle_configuration" "telemetry" {
  bucket = aws_s3_bucket.telemetry.id
  rule {
    id     = "expire-demo-output"
    status = "Enabled"
    filter {}
    expiration {
      days = 7
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}
