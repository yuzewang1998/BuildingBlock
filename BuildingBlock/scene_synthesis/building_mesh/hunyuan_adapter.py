"""Shell adapter for official Hunyuan3D-Omni bbox-control inference.

This module intentionally does not wrap or reimplement Hunyuan internals.  It
defines the process contract used by the downstream BuildingBlock mesh lane:
call an official command, capture every attempt, classify failures, and leave
mesh interpretation to the assembly stage.
"""

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union


DEFAULT_CONCURRENCY = 1
DEFAULT_TIMEOUT_SECONDS = 20 * 60
DEFAULT_RETRY_COUNT = 1

FAILURE_TIMEOUT = "process_timeout"
FAILURE_NONZERO_EXIT = "nonzero_exit_without_output_mesh"
FAILURE_MISSING_OUTPUT = "missing_output_mesh"
FAILURE_INVALID_CONTRACT = "invalid_contract_payload"
FAILURE_UNSUPPORTED_PROMPT = "unsupported_prompt_or_class"
FAILURE_NORMALIZATION = "mesh_normalization_failure"

RETRYABLE_FAILURES = {
    FAILURE_TIMEOUT,
    FAILURE_NONZERO_EXIT,
    FAILURE_MISSING_OUTPUT,
}

NON_RETRYABLE_FAILURES = {
    FAILURE_INVALID_CONTRACT,
    FAILURE_UNSUPPORTED_PROMPT,
    FAILURE_NORMALIZATION,
}

DISCOVERED_MESH_EXTENSIONS = (".obj",)


@dataclass(frozen=True)
class HunyuanAdapterPolicy:
    """Runtime policy for one-part-at-a-time V1 inference."""

    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    retry_count: int = DEFAULT_RETRY_COUNT
    concurrency: int = DEFAULT_CONCURRENCY

    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")


@dataclass
class HunyuanAttempt:
    """JSON-serializable capture for one official inference subprocess run."""

    attempt_number: int
    start_time: str
    end_time: str
    command: List[str]
    exit_code: Optional[int]
    timeout: bool
    failure_type: Optional[str]
    retryable: bool
    stdout_path: str
    stderr_path: str
    discovered_output_paths: List[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.failure_type is None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class HunyuanPartResult:
    """Outcome for a single contract part after all allowed attempts."""

    layout_id: str
    schema_version: str
    contract_path: str
    part_id: str
    source_actor_label: Optional[str]
    target_class: Optional[str]
    attempts: List[HunyuanAttempt]
    raw_output_path: Optional[str]
    lifecycle_states: List[str]

    @property
    def succeeded(self) -> bool:
        return self.raw_output_path is not None

    @property
    def final_failure_type(self) -> Optional[str]:
        for attempt in reversed(self.attempts):
            if attempt.failure_type is not None:
                return attempt.failure_type
        return None

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return payload


Runner = Callable[..., subprocess.CompletedProcess]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_contract_part(contract_part: Union[str, Path, Mapping[str, object]]) -> Dict[str, object]:
    if isinstance(contract_part, Mapping):
        return dict(contract_part)
    path = Path(contract_part)
    with path.open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("contract payload must be a JSON object")
    return payload


def validate_contract_part(contract_part: Mapping[str, object]) -> None:
    required = (
        "schema_version",
        "layout_id",
        "part_id",
        "target_prompt",
        "target_class",
        "bbox",
        "contract_path",
    )
    missing = [field_name for field_name in required if field_name not in contract_part]
    if missing:
        raise ValueError("contract payload missing required fields: {}".format(", ".join(missing)))

    bbox = contract_part.get("bbox")
    if not isinstance(bbox, Mapping):
        raise ValueError("contract bbox must be an object")
    for field_name in ("center", "size"):
        value = bbox.get(field_name)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
            raise ValueError("contract bbox.{} must be a 3-value sequence".format(field_name))
        for number in value:
            float(number)


def classify_preflight_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "prompt" in message or "class" in message or "unsupported" in message:
        return FAILURE_UNSUPPORTED_PROMPT
    return FAILURE_INVALID_CONTRACT


def classify_stderr_failure(stderr: Union[str, bytes, None]) -> Optional[str]:
    if isinstance(stderr, bytes):
        message = stderr.decode("utf-8", errors="replace").lower()
    else:
        message = (stderr or "").lower()
    if "unsupported" in message and ("prompt" in message or "class" in message):
        return FAILURE_UNSUPPORTED_PROMPT
    return None


def classify_process_failure(
    *,
    timed_out: bool,
    exit_code: Optional[int],
    discovered_output_paths: Sequence[str],
    stderr: Union[str, bytes, None] = None,
) -> Optional[str]:
    if timed_out:
        return FAILURE_TIMEOUT
    stderr_failure = classify_stderr_failure(stderr)
    if stderr_failure is not None:
        return stderr_failure
    if exit_code != 0 and not discovered_output_paths:
        return FAILURE_NONZERO_EXIT
    if not discovered_output_paths:
        return FAILURE_MISSING_OUTPUT
    return None


def is_retryable_failure(failure_type: Optional[str]) -> bool:
    return failure_type in RETRYABLE_FAILURES


def build_hunyuan_command(
    command_template: Union[str, Sequence[str]],
    *,
    contract_part: Mapping[str, object],
    output_dir: Union[str, Path],
) -> List[str]:
    reference_image = str(
        contract_part.get("reference_image_path")
        or contract_part.get("reference_image")
        or ""
    )
    values = {
        "contract_path": str(contract_part["contract_path"]),
        "output_dir": str(output_dir),
        "part_id": str(contract_part["part_id"]),
        "target_prompt": str(contract_part.get("target_prompt", "")),
        "target_class": str(contract_part.get("target_class", "")),
        "reference_image": reference_image,
        "reference_image_path": reference_image,
        "image": reference_image,
        "bbox_size_json": json.dumps(contract_part.get("bbox", {}).get("size", [])),
        "bbox_sx": str(contract_part.get("bbox", {}).get("size", [None, None, None])[0]),
        "bbox_sy": str(contract_part.get("bbox", {}).get("size", [None, None, None])[1]),
        "bbox_sz": str(contract_part.get("bbox", {}).get("size", [None, None, None])[2]),
    }

    if isinstance(command_template, str):
        rendered = Template(command_template).safe_substitute(values)
        command = shlex.split(rendered)
    else:
        command = [Template(str(token)).safe_substitute(values) for token in command_template]

    return repair_missing_image_argument(command, reference_image)


def repair_missing_image_argument(command: Sequence[str], reference_image: str) -> List[str]:
    """Fill a bare/empty --image argument from the contract reference image.

    This makes the adapter tolerant of shell-expanded templates such as
    ``--hunyuan-command "... --image $reference_image ..."`` where the outer
    shell expands ``$reference_image`` before Python receives the template.
    """
    if not reference_image:
        return list(command)

    repaired: List[str] = []
    index = 0
    while index < len(command):
        token = str(command[index])
        if token == "--image":
            repaired.append(token)
            missing_value = index + 1 >= len(command) or str(command[index + 1]).startswith("-")
            if missing_value:
                repaired.append(reference_image)
            index += 1
            continue
        if token == "--image=":
            repaired.append("--image={}".format(reference_image))
            index += 1
            continue

        repaired.append(token)
        index += 1

    return repaired


def discover_output_meshes(output_dir: Union[str, Path], part_id: str) -> List[str]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    matches = []
    for path in output_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in DISCOVERED_MESH_EXTENSIONS:
            if part_id in path.stem or part_id in str(path.parent):
                matches.append(str(path))
    if not matches:
        for path in output_path.iterdir():
            if path.is_file() and path.suffix.lower() in DISCOVERED_MESH_EXTENSIONS:
                matches.append(str(path))
    return sorted(matches)


def write_attempt_streams(
    *,
    logs_dir: Union[str, Path],
    part_id: str,
    attempt_number: int,
    stdout: Union[str, bytes, None],
    stderr: Union[str, bytes, None],
) -> Dict[str, str]:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    prefix = "{}__attempt_{:02d}".format(part_id, attempt_number)
    stdout_path = logs_path / "{}.stdout.txt".format(prefix)
    stderr_path = logs_path / "{}.stderr.txt".format(prefix)

    stdout_text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout or ""
    stderr_text = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr or ""
    stdout_path.write_text(stdout_text)
    stderr_path.write_text(stderr_text)
    return {"stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}


class HunyuanAdapter:
    """Runs official Hunyuan inference for contract parts with V1 policy."""

    def __init__(
        self,
        command_template: Union[str, Sequence[str]],
        output_root: Union[str, Path],
        policy: Optional[HunyuanAdapterPolicy] = None,
        runner: Optional[Runner] = None,
    ):
        self.command_template = command_template
        self.output_root = Path(output_root)
        self.policy = policy or HunyuanAdapterPolicy()
        self.runner = runner or subprocess.run

    def run_part(self, contract_part: Union[str, Path, Mapping[str, object]]) -> HunyuanPartResult:
        attempts = []

        try:
            payload = load_contract_part(contract_part)
            validate_contract_part(payload)
        except Exception as exc:
            payload = contract_part if isinstance(contract_part, Mapping) else {}
            layout_id = str(payload.get("layout_id", "")) if isinstance(payload, Mapping) else ""
            schema_version = (
                str(payload.get("schema_version", "")) if isinstance(payload, Mapping) else ""
            )
            contract_path = (
                str(payload.get("contract_path", contract_part))
                if isinstance(payload, Mapping)
                else str(contract_part)
            )
            part_id = (
                str(payload.get("part_id", "unknown_part"))
                if isinstance(payload, Mapping)
                else "unknown_part"
            )
            source_actor_label = (
                payload.get("source_actor_label") if isinstance(payload, Mapping) else None
            )
            target_class = payload.get("target_class") if isinstance(payload, Mapping) else None
            logs_dir = self.output_root / part_id / "logs"
            streams = write_attempt_streams(
                logs_dir=logs_dir,
                part_id=part_id,
                attempt_number=1,
                stdout="",
                stderr=str(exc),
            )
            attempt = HunyuanAttempt(
                attempt_number=1,
                start_time=utc_now_iso(),
                end_time=utc_now_iso(),
                command=[],
                exit_code=None,
                timeout=False,
                failure_type=classify_preflight_failure(exc),
                retryable=False,
                stdout_path=streams["stdout_path"],
                stderr_path=streams["stderr_path"],
                discovered_output_paths=[],
            )
            return HunyuanPartResult(
                layout_id=layout_id,
                schema_version=schema_version,
                contract_path=contract_path,
                part_id=part_id,
                source_actor_label=str(source_actor_label) if source_actor_label is not None else None,
                target_class=str(target_class) if target_class is not None else None,
                attempts=[attempt],
                raw_output_path=None,
                lifecycle_states=["discovered", "contract_emitted", "hunyuan_failed"],
            )

        layout_id = str(payload.get("layout_id", ""))
        schema_version = str(payload.get("schema_version", ""))
        contract_path = str(payload.get("contract_path", contract_part))
        part_id = str(payload.get("part_id", "unknown_part"))
        source_actor_label = payload.get("source_actor_label")
        target_class = payload.get("target_class")
        part_output_dir = self.output_root / part_id
        logs_dir = part_output_dir / "logs"
        part_output_dir.mkdir(parents=True, exist_ok=True)
        total_attempts = self.policy.retry_count + 1
        raw_output_path = None

        for attempt_number in range(1, total_attempts + 1):
            start_time = utc_now_iso()
            command = build_hunyuan_command(
                self.command_template,
                contract_part=payload,
                output_dir=part_output_dir,
            )
            exit_code = None
            timed_out = False
            stdout = ""
            stderr = ""
            try:
                completed = self.runner(
                    command,
                    cwd=None,
                    capture_output=True,
                    text=True,
                    timeout=self.policy.timeout_seconds,
                    check=False,
                )
                exit_code = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = exc.stdout
                stderr = exc.stderr or "process exceeded timeout_seconds={}".format(
                    self.policy.timeout_seconds
                )

            discovered_output_paths = discover_output_meshes(part_output_dir, part_id)
            failure_type = classify_process_failure(
                timed_out=timed_out,
                exit_code=exit_code,
                discovered_output_paths=discovered_output_paths,
                stderr=stderr,
            )
            streams = write_attempt_streams(
                logs_dir=logs_dir,
                part_id=part_id,
                attempt_number=attempt_number,
                stdout=stdout,
                stderr=stderr,
            )
            attempt = HunyuanAttempt(
                attempt_number=attempt_number,
                start_time=start_time,
                end_time=utc_now_iso(),
                command=command,
                exit_code=exit_code,
                timeout=timed_out,
                failure_type=failure_type,
                retryable=is_retryable_failure(failure_type),
                stdout_path=streams["stdout_path"],
                stderr_path=streams["stderr_path"],
                discovered_output_paths=discovered_output_paths,
            )
            attempts.append(attempt)

            if failure_type is None:
                raw_output_path = discovered_output_paths[0]
                break
            if not is_retryable_failure(failure_type):
                break

        return HunyuanPartResult(
            layout_id=layout_id,
            schema_version=schema_version,
            contract_path=contract_path,
            part_id=part_id,
            source_actor_label=str(source_actor_label) if source_actor_label is not None else None,
            target_class=str(target_class) if target_class is not None else None,
            attempts=attempts,
            raw_output_path=raw_output_path,
            lifecycle_states=[
                "discovered",
                "contract_emitted",
                "submitted_to_hunyuan",
                "hunyuan_succeeded" if raw_output_path else "hunyuan_failed",
            ],
        )

    def run_many(
        self,
        contract_parts: Iterable[Union[str, Path, Mapping[str, object]]],
    ) -> List[HunyuanPartResult]:
        if self.policy.concurrency != 1:
            raise NotImplementedError(
                "V1 adapter defaults to sequential concurrency=1; parallel execution is not implemented"
            )
        return [self.run_part(contract_part) for contract_part in contract_parts]
