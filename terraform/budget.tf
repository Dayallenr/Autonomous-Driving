# The ground-rules $1 budget alarm (CLAUDE.md: zero spend by default). This
# is the safety net behind every real-AWS step: issue #18's wizard applies it
# first and refuses to create the SQS queues until the Budgets API confirms
# it exists. A notification-only budget costs nothing — AWS charges only for
# budget *actions*, and this one has none.
#
# AWS Budgets is a global service; the provider's region is irrelevant here.

resource "aws_budgets_budget" "zero_spend" {
  name         = "${var.cluster_name}-zero-spend"
  budget_type  = "COST"
  limit_amount = "1.0"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # 1% of $1 = one actual cent: this project is meant to spend nothing, so
  # any real spend at all is alarm-worthy, not just approaching the limit.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 1
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_notification_email]
  }

  # And an early warning when the month's trajectory would blow the $1 cap.
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_notification_email]
  }
}
