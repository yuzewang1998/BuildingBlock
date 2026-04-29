"""Manifest and failure reporting for BuildingBlock layout-to-mesh V1."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Union

from .hunyuan_adapter import FAILURE_NORMALIZATION


REQUIRED_MANIFEST_FIELDS = (
    "layout_id",
    "schema_version",
    "contract_path",
    "part_id",
    "source_actor_label",
    "target_class",
    "attempts",
    "raw_output_path",
    "normalized_output_path",
    "placeholder_output_path",
    "raw_assembly_path",
    "placeholder_assembly_path",
)

REQUIRED_FAILURE_FIELDS = (
    "layout_id",
    "schema_version",
    "part_id",
    "attempt_number",
    "failure_type",
    "exit_code",
    "timeout",
    "stderr_path",
)

def _as_dict(value: object) -> Dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    raise TypeError("cannot convert value to dict: {!r}".format(value))


def _attempts_to_dicts(attempts: Iterable[object]) -> List[Dict[str, object]]:
    return [_as_dict(attempt) for attempt in attempts]


def _part_result_value(result: object, key: str, default=None):
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _assembly_part_map(assembly_result: object) -> Dict[str, Dict[str, object]]:
    if assembly_result is None:
        return {}
    payload = _as_dict(assembly_result)
    parts = payload.get("parts", [])
    mapped = {}
    for part in parts:
        part_payload = _as_dict(part)
        mapped[str(part_payload["part_id"])] = part_payload
    return mapped


def build_manifest(
    hunyuan_results: Iterable[object],
    assembly_result: object,
) -> Dict[str, object]:
    assembly_payload = _as_dict(assembly_result)
    parts_by_id = _assembly_part_map(assembly_result)
    entries = []

    for result in hunyuan_results:
        part_id = str(_part_result_value(result, "part_id"))
        assembly_part = parts_by_id.get(part_id, {})
        entry = {
            "layout_id": _part_result_value(result, "layout_id"),
            "schema_version": _part_result_value(result, "schema_version"),
            "contract_path": _part_result_value(result, "contract_path"),
            "part_id": part_id,
            "source_actor_label": _part_result_value(result, "source_actor_label"),
            "target_class": _part_result_value(result, "target_class"),
            "attempts": _attempts_to_dicts(_part_result_value(result, "attempts", [])),
            "raw_output_path": _part_result_value(result, "raw_output_path"),
            "normalized_output_path": assembly_part.get("normalized_output_path"),
            "placeholder_output_path": assembly_part.get("placeholder_output_path"),
            "raw_assembly_path": assembly_payload.get("raw_assembly_path"),
            "placeholder_assembly_path": assembly_payload.get("placeholder_assembly_path"),
            "lifecycle_states": assembly_part.get(
                "lifecycle_states", _part_result_value(result, "lifecycle_states", [])
            ),
        }
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in entry]
        if missing:
            raise ValueError("manifest entry missing fields: {}".format(", ".join(missing)))
        entries.append(entry)

    return {
        "layout_id": assembly_payload.get("layout_id"),
        "schema_version": assembly_payload.get("schema_version"),
        "raw_assembly_path": assembly_payload.get("raw_assembly_path"),
        "placeholder_assembly_path": assembly_payload.get("placeholder_assembly_path"),
        "parts": entries,
    }


def build_failures(hunyuan_results: Iterable[object]) -> Dict[str, object]:
    failures = []
    layout_id = None
    schema_version = None

    for result in hunyuan_results:
        layout_id = layout_id or _part_result_value(result, "layout_id")
        schema_version = schema_version or _part_result_value(result, "schema_version")
        part_id = _part_result_value(result, "part_id")
        attempts = _attempts_to_dicts(_part_result_value(result, "attempts", []))
        lifecycle_states = _part_result_value(result, "lifecycle_states", [])
        for attempt in attempts:
            failure_type = attempt.get("failure_type")
            if failure_type is None:
                continue
            failure = {
                "layout_id": _part_result_value(result, "layout_id"),
                "schema_version": _part_result_value(result, "schema_version"),
                "part_id": part_id,
                "attempt_number": attempt.get("attempt_number"),
                "failure_type": failure_type,
                "exit_code": attempt.get("exit_code"),
                "timeout": attempt.get("timeout"),
                "stderr_path": attempt.get("stderr_path"),
            }
            missing = [field for field in REQUIRED_FAILURE_FIELDS if field not in failure]
            if missing:
                raise ValueError("failure entry missing fields: {}".format(", ".join(missing)))
            failures.append(failure)

        if FAILURE_NORMALIZATION in lifecycle_states and attempts:
            final_attempt = attempts[-1]
            failures.append(
                {
                    "layout_id": _part_result_value(result, "layout_id"),
                    "schema_version": _part_result_value(result, "schema_version"),
                    "part_id": part_id,
                    "attempt_number": final_attempt.get("attempt_number"),
                    "failure_type": FAILURE_NORMALIZATION,
                    "exit_code": final_attempt.get("exit_code"),
                    "timeout": final_attempt.get("timeout"),
                    "stderr_path": final_attempt.get("stderr_path"),
                }
            )

    return {
        "layout_id": layout_id,
        "schema_version": schema_version,
        "failures": failures,
    }


def add_assembly_failures(
    failures_payload: Dict[str, object],
    hunyuan_results: Iterable[object],
    assembly_result: object,
) -> Dict[str, object]:
    assembly_parts = _assembly_part_map(assembly_result)
    results_by_part_id = {
        str(_part_result_value(result, "part_id")): result for result in hunyuan_results
    }
    failures = list(failures_payload.get("failures", []))

    for part_id, assembly_part in assembly_parts.items():
        lifecycle_states = assembly_part.get("lifecycle_states", [])
        if FAILURE_NORMALIZATION not in lifecycle_states:
            continue
        result = results_by_part_id.get(part_id)
        attempts = _attempts_to_dicts(_part_result_value(result, "attempts", []))
        if not attempts:
            continue
        final_attempt = attempts[-1]
        failures.append(
            {
                "layout_id": _part_result_value(result, "layout_id"),
                "schema_version": _part_result_value(result, "schema_version"),
                "part_id": part_id,
                "attempt_number": final_attempt.get("attempt_number"),
                "failure_type": FAILURE_NORMALIZATION,
                "exit_code": final_attempt.get("exit_code"),
                "timeout": final_attempt.get("timeout"),
                "stderr_path": final_attempt.get("stderr_path"),
            }
        )

    return {
        "layout_id": failures_payload.get("layout_id"),
        "schema_version": failures_payload.get("schema_version"),
        "failures": failures,
    }


def write_json_report(payload: Mapping[str, object], path: Union[str, Path]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(output_path)


def write_reports(
    hunyuan_results: Iterable[object],
    assembly_result: object,
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    results = list(hunyuan_results)
    output_path = Path(output_dir)
    manifest = build_manifest(results, assembly_result)
    failures = add_assembly_failures(build_failures(results), results, assembly_result)
    return {
        "manifest_path": write_json_report(manifest, output_path / "manifest.json"),
        "failures_path": write_json_report(failures, output_path / "failures.json"),
    }
