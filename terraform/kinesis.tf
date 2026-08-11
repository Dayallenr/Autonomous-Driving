# On-demand mode rather than provisioned shards: this stream exists for one
# short telemetry demo (see scripts/run_worker.py --telemetry-backend
# kinesis), and on-demand bills per request instead of per shard-hour
# regardless of whether anything is actively streaming — the right shape for
# a stream that's up for under an hour. It's also the resource this project's
# cost-traps checklist flags explicitly: billed from creation to deletion,
# not from first use.
resource "aws_kinesis_stream" "telemetry" {
  name = "${var.cluster_name}-telemetry"
  stream_mode_details {
    stream_mode = "ON_DEMAND"
  }
  retention_period = var.kinesis_retention_hours

  encryption_type = "KMS"
  kms_key_id      = "alias/aws/kinesis" # AWS-managed key — a demo stream that lives for an hour doesn't need a dedicated CMK
}
