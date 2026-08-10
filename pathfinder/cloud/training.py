"""
Managed training jobs: local subprocess or SageMaker.

What this abstraction is for
----------------------------
SageMaker's actual contract is small and worth taking seriously, because
matching it is what makes a training script portable:

* Input data is **materialized into the container** at a known path
  (``/opt/ml/input/data/<channel>``) before the script starts. The script reads
  files, never S3 URIs. This is why a SageMaker script runs unchanged locally.
* Hyperparameters arrive as a **JSON file** at
  ``/opt/ml/input/config/hyperparameters.json``.
* Model artifacts are written to ``/opt/ml/model``, and everything there is
  tarred and uploaded when the job exits cleanly.
* Exit code 0 means success. Anything else fails the job.

The local backend honours all four against a temporary directory. A script
written for one backend therefore runs on the other with no changes — which is
the only property that makes the abstraction worth having. An abstraction that
merely renames ``subprocess.run`` to ``fit()`` would buy nothing.

Spot instances
--------------
The SageMaker backend defaults to managed spot training, which is roughly 70%
cheaper and is the right default for a job that checkpoints. It requires
``max_wait >= max_run``, and getting that wrong is a job that fails at submit
time with an unhelpful message — so it is validated here instead.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pathfinder.cloud.objects import DatasetManifest, DatasetRegistry

logger = logging.getLogger(__name__)

__all__ = [
    "JobStatus",
    "LocalTrainingJobRunner",
    "SageMakerTrainingJobRunner",
    "TrainingJob",
    "TrainingJobRunner",
    "build_runner",
]

#: SageMaker's container filesystem contract. Mirrored exactly by the local
#: backend so a training script cannot tell the two apart.
SM_INPUT_DIR = "/opt/ml/input/data"
SM_CONFIG_DIR = "/opt/ml/input/config"
SM_MODEL_DIR = "/opt/ml/model"
SM_OUTPUT_DIR = "/opt/ml/output"


class JobStatus:
    PENDING = "Pending"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"


@dataclass
class TrainingJob:
    """A submitted job and its outcome."""

    name: str
    status: str
    started_at: str
    #: Dataset version consumed. Recorded so "what data produced these weights?"
    #: is answerable from the job record alone.
    dataset_name: str = ""
    dataset_version: str = ""
    hyperparameters: dict = field(default_factory=dict)
    model_artifacts: str = ""
    seconds_elapsed: float = 0.0
    exit_code: int | None = None
    log_tail: list[str] = field(default_factory=list)
    failure_reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == JobStatus.COMPLETED

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "dataset": f"{self.dataset_name}@{self.dataset_version}"
            if self.dataset_name
            else "",
            "hyperparameters": self.hyperparameters,
            "model_artifacts": self.model_artifacts,
            "seconds_elapsed": round(self.seconds_elapsed, 2),
            "exit_code": self.exit_code,
            "failure_reason": self.failure_reason,
        }


class TrainingJobRunner(ABC):
    """Submit-and-wait interface for a managed training job."""

    @abstractmethod
    def fit(
        self,
        *,
        job_name: str,
        entry_point: Path | str,
        dataset: DatasetManifest | None = None,
        hyperparameters: dict | None = None,
        channel: str = "training",
        timeout_seconds: float = 3600.0,
    ) -> TrainingJob:
        """Run one training job to completion."""


class LocalTrainingJobRunner(TrainingJobRunner):
    """Runs the entry point as a subprocess against a SageMaker-shaped directory.

    Deliberately a subprocess rather than an in-process import: a training
    script that mutates global torch state, seeds RNGs, or calls ``sys.exit``
    must not be able to affect the orchestrator. Process isolation also makes
    the exit-code contract real instead of simulated.
    """

    def __init__(
        self,
        registry: DatasetRegistry | None = None,
        *,
        workspace: Path | str | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.registry = registry
        self.workspace = Path(workspace) if workspace else None
        self.python_executable = python_executable or sys.executable

    def fit(
        self,
        *,
        job_name: str,
        entry_point: Path | str,
        dataset: DatasetManifest | None = None,
        hyperparameters: dict | None = None,
        channel: str = "training",
        timeout_seconds: float = 3600.0,
    ) -> TrainingJob:
        entry_point = Path(entry_point)
        if not entry_point.exists():
            raise FileNotFoundError(f"training entry point not found: {entry_point}")

        job = TrainingJob(
            name=job_name,
            status=JobStatus.IN_PROGRESS,
            started_at=datetime.now(UTC).isoformat(),
            dataset_name=dataset.name if dataset else "",
            dataset_version=dataset.version if dataset else "",
            hyperparameters=hyperparameters or {},
        )

        root = Path(self.workspace) if self.workspace else Path(
            tempfile.mkdtemp(prefix=f"sm-{job_name}-")
        )
        input_dir = root / "input" / "data" / channel
        config_dir = root / "input" / "config"
        model_dir = root / "model"
        for directory in (input_dir, config_dir, model_dir, root / "output"):
            directory.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        try:
            # Materialize the dataset into the container path, exactly as
            # SageMaker does before the script runs.
            if dataset is not None:
                if self.registry is None:
                    raise ValueError("a DatasetRegistry is required to materialize a dataset")
                self.registry.download(dataset, input_dir)
                logger.info(
                    "materialized %s@%s (%d files) into %s",
                    dataset.name, dataset.version, dataset.file_count, input_dir,
                )

            (config_dir / "hyperparameters.json").write_text(
                json.dumps(hyperparameters or {}, indent=2), encoding="utf-8"
            )

            environment = {
                "SM_CHANNEL_TRAINING": str(input_dir),
                "SM_MODEL_DIR": str(model_dir),
                "SM_OUTPUT_DIR": str(root / "output"),
                "SM_HYPERPARAMETERS": str(config_dir / "hyperparameters.json"),
                "PATHFINDER_DATASET_VERSION": job.dataset_version,
            }
            import os

            result = subprocess.run(
                [self.python_executable, str(entry_point)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ, **environment},
            )
            job.exit_code = result.returncode
            # Keep a bounded tail: a chatty training script can emit megabytes,
            # and holding all of it in the job record is a memory leak.
            job.log_tail = (result.stdout + result.stderr).splitlines()[-40:]

            if result.returncode == 0:
                job.status = JobStatus.COMPLETED
                job.model_artifacts = str(model_dir)
            else:
                job.status = JobStatus.FAILED
                job.failure_reason = f"entry point exited {result.returncode}"

        except subprocess.TimeoutExpired:
            job.status = JobStatus.FAILED
            job.failure_reason = f"exceeded timeout of {timeout_seconds}s"
        except Exception as error:
            job.status = JobStatus.FAILED
            job.failure_reason = str(error)
        finally:
            job.seconds_elapsed = time.perf_counter() - started

        logger.info("job %s finished: %s (%.1fs)", job_name, job.status, job.seconds_elapsed)
        return job


class SageMakerTrainingJobRunner(TrainingJobRunner):
    """Amazon SageMaker backend using managed spot training."""

    def __init__(
        self,
        *,
        role_arn: str,
        image_uri: str,
        output_path: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        use_spot: bool = True,
        max_run_seconds: int = 3600,
        max_wait_seconds: int = 7200,
        region: str = "us-east-1",
        endpoint_url: str | None = None,
    ) -> None:
        import boto3

        if use_spot and max_wait_seconds < max_run_seconds:
            # SageMaker rejects this at submit time with an opaque message.
            # Catching it here turns a confusing API error into a clear one.
            raise ValueError(
                f"managed spot training requires max_wait_seconds "
                f"({max_wait_seconds}) >= max_run_seconds ({max_run_seconds}): "
                "max_wait covers run time plus time spent waiting for capacity"
            )
        self.role_arn = role_arn
        self.image_uri = image_uri
        self.output_path = output_path
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.use_spot = use_spot
        self.max_run_seconds = max_run_seconds
        self.max_wait_seconds = max_wait_seconds
        self._client = boto3.client(
            "sagemaker", region_name=region, endpoint_url=endpoint_url
        )

    def fit(
        self,
        *,
        job_name: str,
        entry_point: Path | str,
        dataset: DatasetManifest | None = None,
        hyperparameters: dict | None = None,
        channel: str = "training",
        timeout_seconds: float = 3600.0,
    ) -> TrainingJob:
        if dataset is None:
            raise ValueError("SageMaker training requires a dataset to mount as a channel")

        job = TrainingJob(
            name=job_name,
            status=JobStatus.IN_PROGRESS,
            started_at=datetime.now(UTC).isoformat(),
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            hyperparameters=hyperparameters or {},
        )

        request: dict = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": self.image_uri,
                "TrainingInputMode": "File",
            },
            "RoleArn": self.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": channel,
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            # Pinned to the immutable version prefix, never to a
                            # mutable "latest" path.
                            "S3Uri": f"{self.output_path.rstrip('/')}/datasets/"
                                     f"{dataset.name}/versions/{dataset.version}/data/",
                            "S3DataDistributionType": "FullyReplicated",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": self.output_path},
            "ResourceConfig": {
                "InstanceType": self.instance_type,
                "InstanceCount": self.instance_count,
                "VolumeSizeInGB": 50,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": self.max_run_seconds},
            "HyperParameters": {
                key: str(value) for key, value in (hyperparameters or {}).items()
            },
        }
        if self.use_spot:
            request["EnableManagedSpotTraining"] = True
            request["StoppingCondition"]["MaxWaitTimeInSeconds"] = self.max_wait_seconds
            # Spot instances are interrupted; without a checkpoint path the job
            # restarts from scratch and the cost saving evaporates.
            request["CheckpointConfig"] = {
                "S3Uri": f"{self.output_path.rstrip('/')}/checkpoints/{job_name}/"
            }

        started = time.perf_counter()
        try:
            self._client.create_training_job(**request)
            waiter = self._client.get_waiter("training_job_completed_or_stopped")
            waiter.wait(
                TrainingJobName=job_name,
                WaiterConfig={"Delay": 30, "MaxAttempts": int(timeout_seconds // 30) or 1},
            )
            description = self._client.describe_training_job(TrainingJobName=job_name)
            job.status = (
                JobStatus.COMPLETED
                if description["TrainingJobStatus"] == "Completed"
                else JobStatus.FAILED
            )
            job.model_artifacts = description.get("ModelArtifacts", {}).get(
                "S3ModelArtifacts", ""
            )
            job.failure_reason = description.get("FailureReason", "")
        except Exception as error:
            job.status = JobStatus.FAILED
            job.failure_reason = str(error)
        finally:
            job.seconds_elapsed = time.perf_counter() - started
        return job


def build_runner(
    backend: str,
    *,
    registry: DatasetRegistry | None = None,
    workspace: Path | str | None = None,
    role_arn: str = "",
    image_uri: str = "",
    output_path: str = "",
    **kwargs,
) -> TrainingJobRunner:
    """Construct the configured training backend.

    Raises:
        ValueError: On an unknown backend, or a sagemaker backend missing
            required configuration.
    """
    normalized = backend.strip().lower()
    if normalized == "local":
        return LocalTrainingJobRunner(registry, workspace=workspace)
    if normalized == "sagemaker":
        missing = [
            name
            for name, value in (
                ("role_arn", role_arn),
                ("image_uri", image_uri),
                ("output_path", output_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"sagemaker backend requires: {', '.join(missing)}")
        return SageMakerTrainingJobRunner(
            role_arn=role_arn, image_uri=image_uri, output_path=output_path, **kwargs
        )
    raise ValueError(f"unknown training backend {backend!r}; expected local or sagemaker")
