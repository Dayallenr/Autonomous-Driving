from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EpisodeSpec(_message.Message):
    __slots__ = ("episode_id", "town", "weather", "route_length_m", "seed", "max_steps", "delta_seconds", "traffic_density", "pedestrian_density")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    TOWN_FIELD_NUMBER: _ClassVar[int]
    WEATHER_FIELD_NUMBER: _ClassVar[int]
    ROUTE_LENGTH_M_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    MAX_STEPS_FIELD_NUMBER: _ClassVar[int]
    DELTA_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_DENSITY_FIELD_NUMBER: _ClassVar[int]
    PEDESTRIAN_DENSITY_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    town: str
    weather: str
    route_length_m: float
    seed: int
    max_steps: int
    delta_seconds: float
    traffic_density: float
    pedestrian_density: float
    def __init__(self, episode_id: _Optional[str] = ..., town: _Optional[str] = ..., weather: _Optional[str] = ..., route_length_m: _Optional[float] = ..., seed: _Optional[int] = ..., max_steps: _Optional[int] = ..., delta_seconds: _Optional[float] = ..., traffic_density: _Optional[float] = ..., pedestrian_density: _Optional[float] = ...) -> None: ...

class InfractionCount(_message.Message):
    __slots__ = ("kind", "count")
    KIND_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    kind: str
    count: int
    def __init__(self, kind: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class EpisodeResult(_message.Message):
    __slots__ = ("episode_id", "worker_id", "route_completion", "infraction_penalty", "driving_score", "infractions", "distance_travelled_m", "route_length_m", "frames", "duration_seconds", "mean_fps", "status", "termination_reason", "model_version", "dataset_version", "simulator_backend")
    EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ROUTE_COMPLETION_FIELD_NUMBER: _ClassVar[int]
    INFRACTION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    DRIVING_SCORE_FIELD_NUMBER: _ClassVar[int]
    INFRACTIONS_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_TRAVELLED_M_FIELD_NUMBER: _ClassVar[int]
    ROUTE_LENGTH_M_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    MEAN_FPS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TERMINATION_REASON_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    DATASET_VERSION_FIELD_NUMBER: _ClassVar[int]
    SIMULATOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    episode_id: str
    worker_id: str
    route_completion: float
    infraction_penalty: float
    driving_score: float
    infractions: _containers.RepeatedCompositeFieldContainer[InfractionCount]
    distance_travelled_m: float
    route_length_m: float
    frames: int
    duration_seconds: float
    mean_fps: float
    status: str
    termination_reason: str
    model_version: str
    dataset_version: str
    simulator_backend: str
    def __init__(self, episode_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., route_completion: _Optional[float] = ..., infraction_penalty: _Optional[float] = ..., driving_score: _Optional[float] = ..., infractions: _Optional[_Iterable[_Union[InfractionCount, _Mapping]]] = ..., distance_travelled_m: _Optional[float] = ..., route_length_m: _Optional[float] = ..., frames: _Optional[int] = ..., duration_seconds: _Optional[float] = ..., mean_fps: _Optional[float] = ..., status: _Optional[str] = ..., termination_reason: _Optional[str] = ..., model_version: _Optional[str] = ..., dataset_version: _Optional[str] = ..., simulator_backend: _Optional[str] = ...) -> None: ...

class RegisterRequest(_message.Message):
    __slots__ = ("worker_id", "hostname", "simulator_backend", "carla_port")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    SIMULATOR_BACKEND_FIELD_NUMBER: _ClassVar[int]
    CARLA_PORT_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    hostname: str
    simulator_backend: str
    carla_port: int
    def __init__(self, worker_id: _Optional[str] = ..., hostname: _Optional[str] = ..., simulator_backend: _Optional[str] = ..., carla_port: _Optional[int] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("accepted", "run_id", "heartbeat_interval_seconds", "reason")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_INTERVAL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    run_id: str
    heartbeat_interval_seconds: float
    reason: str
    def __init__(self, accepted: _Optional[bool] = ..., run_id: _Optional[str] = ..., heartbeat_interval_seconds: _Optional[float] = ..., reason: _Optional[str] = ...) -> None: ...

class HeartbeatRequest(_message.Message):
    __slots__ = ("worker_id", "current_episode_id", "frames_completed", "episode_progress", "episodes_completed")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    FRAMES_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    EPISODE_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    EPISODES_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    current_episode_id: str
    frames_completed: int
    episode_progress: float
    episodes_completed: int
    def __init__(self, worker_id: _Optional[str] = ..., current_episode_id: _Optional[str] = ..., frames_completed: _Optional[int] = ..., episode_progress: _Optional[float] = ..., episodes_completed: _Optional[int] = ...) -> None: ...

class HeartbeatResponse(_message.Message):
    __slots__ = ("should_stop", "reason")
    SHOULD_STOP_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    should_stop: bool
    reason: str
    def __init__(self, should_stop: _Optional[bool] = ..., reason: _Optional[str] = ...) -> None: ...

class SubmitResultRequest(_message.Message):
    __slots__ = ("result",)
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: EpisodeResult
    def __init__(self, result: _Optional[_Union[EpisodeResult, _Mapping]] = ...) -> None: ...

class SubmitResultResponse(_message.Message):
    __slots__ = ("accepted", "duplicate")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    DUPLICATE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    duplicate: bool
    def __init__(self, accepted: _Optional[bool] = ..., duplicate: _Optional[bool] = ...) -> None: ...

class RunStatusRequest(_message.Message):
    __slots__ = ("run_id",)
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    def __init__(self, run_id: _Optional[str] = ...) -> None: ...

class WorkerStatus(_message.Message):
    __slots__ = ("worker_id", "current_episode_id", "episode_progress", "episodes_completed", "seconds_since_heartbeat", "healthy")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    CURRENT_EPISODE_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODE_PROGRESS_FIELD_NUMBER: _ClassVar[int]
    EPISODES_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    SECONDS_SINCE_HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    current_episode_id: str
    episode_progress: float
    episodes_completed: int
    seconds_since_heartbeat: float
    healthy: bool
    def __init__(self, worker_id: _Optional[str] = ..., current_episode_id: _Optional[str] = ..., episode_progress: _Optional[float] = ..., episodes_completed: _Optional[int] = ..., seconds_since_heartbeat: _Optional[float] = ..., healthy: _Optional[bool] = ...) -> None: ...

class RunStatusResponse(_message.Message):
    __slots__ = ("run_id", "episodes_total", "episodes_completed", "episodes_in_flight", "mean_driving_score", "mean_route_completion", "workers", "elapsed_seconds")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    EPISODES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    EPISODES_COMPLETED_FIELD_NUMBER: _ClassVar[int]
    EPISODES_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    MEAN_DRIVING_SCORE_FIELD_NUMBER: _ClassVar[int]
    MEAN_ROUTE_COMPLETION_FIELD_NUMBER: _ClassVar[int]
    WORKERS_FIELD_NUMBER: _ClassVar[int]
    ELAPSED_SECONDS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    episodes_total: int
    episodes_completed: int
    episodes_in_flight: int
    mean_driving_score: float
    mean_route_completion: float
    workers: _containers.RepeatedCompositeFieldContainer[WorkerStatus]
    elapsed_seconds: float
    def __init__(self, run_id: _Optional[str] = ..., episodes_total: _Optional[int] = ..., episodes_completed: _Optional[int] = ..., episodes_in_flight: _Optional[int] = ..., mean_driving_score: _Optional[float] = ..., mean_route_completion: _Optional[float] = ..., workers: _Optional[_Iterable[_Union[WorkerStatus, _Mapping]]] = ..., elapsed_seconds: _Optional[float] = ...) -> None: ...
