# Production-shape SQS: main queue + a real dead-letter queue, matching the
# semantics pathfinder/cloud/queue.py's SqsMessageQueue already implements
# and pathfinder/orchestration.py already depends on (visibility timeout,
# at-least-once delivery, DLQ after max_receive_count). SQS is real AWS even
# in local dev (see CLAUDE.md's zero-spend rule — it has a permanent free
# tier), so this is the one queue resource that isn't LocalStack-only.

resource "aws_sqs_queue" "dead_letter" {
  name                      = "${var.cluster_name}-episodes-dlq"
  message_retention_seconds = 1209600 # 14 days — the maximum, so a poison message is never silently lost
  sqs_managed_sse_enabled   = true
}

resource "aws_sqs_queue" "episodes" {
  name                       = "${var.cluster_name}-episodes"
  visibility_timeout_seconds = var.sqs_visibility_timeout_seconds
  sqs_managed_sse_enabled    = true

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dead_letter.arn
    maxReceiveCount     = var.sqs_max_receive_count
  })
}

# So the DLQ's source can be inspected in the console without cross-
# referencing the redrive_policy JSON by hand.
resource "aws_sqs_queue_redrive_allow_policy" "dead_letter" {
  queue_url = aws_sqs_queue.dead_letter.id
  redrive_allow_policy = jsonencode({
    redrivePermission = "byQueue"
    sourceQueueArns   = [aws_sqs_queue.episodes.arn]
  })
}
