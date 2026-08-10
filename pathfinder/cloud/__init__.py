"""
Cloud service abstractions with local and AWS backends.

Every capability has two implementations behind one interface, selected by
configuration. The local backend is not a mock: it implements the same
semantics (SQS visibility timeouts and DLQ, Kinesis partition ordering,
SageMaker's container filesystem contract, S3 key layout) so that orchestration
logic is genuinely exercised before it reaches AWS.

Pointing the AWS backends at LocalStack runs the real boto3 clients against real
service APIs, which is a stronger test than any hand-written double.
"""
from pathfinder.cloud.objects import (
    DatasetManifest,
    DatasetRegistry,
    LocalObjectStore,
    ObjectStore,
    S3ObjectStore,
    StoredObject,
    build_store,
)
from pathfinder.cloud.queue import (
    LocalMessageQueue,
    Message,
    MessageQueue,
    SqsMessageQueue,
    build_queue,
)
from pathfinder.cloud.stream import (
    KinesisTelemetryStream,
    LocalTelemetryStream,
    StreamRecord,
    TelemetryStream,
    build_stream,
)
from pathfinder.cloud.training import (
    JobStatus,
    LocalTrainingJobRunner,
    SageMakerTrainingJobRunner,
    TrainingJob,
    TrainingJobRunner,
    build_runner,
)
from pathfinder.cloud.warehouse import (
    EPISODE_COLUMNS,
    FRAME_COLUMNS,
    TelemetryWarehouse,
    athena_ddl,
    redshift_ddl,
)

__all__ = [
    "EPISODE_COLUMNS",
    "FRAME_COLUMNS",
    "DatasetManifest",
    "DatasetRegistry",
    "JobStatus",
    "KinesisTelemetryStream",
    "LocalMessageQueue",
    "LocalObjectStore",
    "LocalTelemetryStream",
    "LocalTrainingJobRunner",
    "Message",
    "MessageQueue",
    "ObjectStore",
    "S3ObjectStore",
    "SageMakerTrainingJobRunner",
    "SqsMessageQueue",
    "StoredObject",
    "StreamRecord",
    "TelemetryStream",
    "TelemetryWarehouse",
    "TrainingJob",
    "TrainingJobRunner",
    "athena_ddl",
    "build_queue",
    "build_runner",
    "build_store",
    "build_stream",
    "redshift_ddl",
]
