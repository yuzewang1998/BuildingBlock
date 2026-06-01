"""Minimal mesh assembly utilities for BuildingBlock layout-to-mesh V1."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .hunyuan_adapter import FAILURE_NORMALIZATION


Vector3 = Tuple[float, float, float]


@dataclass
class MeshData:
    vertices: List[Vector3]
    faces: List[Tuple[int, ...]]

    def extend(self, other: "MeshData") -> None:
        offset = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.faces.extend(tuple(index + offset for index in face) for face in other.faces)


@dataclass
class AssemblyPartResult:
    part_id: str
    raw_hunyuan_output_path: Optional[str]
    normalized_output_path: Optional[str]
    placeholder_output_path: Optional[str]
    lifecycle_states: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "part_id": self.part_id,
            "raw_hunyuan_output_path": self.raw_hunyuan_output_path,
            "normalized_output_path": self.normalized_output_path,
            "placeholder_output_path": self.placeholder_output_path,
            "lifecycle_states": list(self.lifecycle_states),
        }


@dataclass
class AssemblyResult:
    layout_id: str
    schema_version: str
    raw_assembly_path: Optional[str]
    placeholder_assembly_path: str
    parts: List[AssemblyPartResult]

    def to_dict(self) -> Dict[str, object]:
        return {
            "layout_id": self.layout_id,
            "schema_version": self.schema_version,
            "raw_assembly_path": self.raw_assembly_path,
            "placeholder_assembly_path": self.placeholder_assembly_path,
            "parts": [part.to_dict() for part in self.parts],
        }


def load_contract_part(contract_part: Union[str, Path, Mapping[str, object]]) -> Dict[str, object]:
    if isinstance(contract_part, Mapping):
        return dict(contract_part)
    path = Path(contract_part)
    with path.open("r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("contract part must be a JSON object")
    return payload


def bbox_center_size(contract_part: Mapping[str, object]) -> Tuple[Vector3, Vector3]:
    bbox = contract_part.get("bbox")
    if not isinstance(bbox, Mapping):
        raise ValueError("contract part {} is missing bbox".format(contract_part.get("part_id")))
    center = tuple(float(value) for value in bbox["center"])
    size = tuple(float(value) for value in bbox["size"])
    if len(center) != 3 or len(size) != 3:
        raise ValueError("bbox center and size must contain exactly three values")
    if any(value <= 0 for value in size):
        raise ValueError("bbox size values must be positive")
    return center, size


def create_box_mesh(center: Sequence[float], size: Sequence[float]) -> MeshData:
    cx, cy, cz = (float(value) for value in center)
    sx, sy, sz = (float(value) for value in size)
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    vertices = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    faces = [
        (1, 2, 3, 4),
        (5, 8, 7, 6),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 8, 4),
        (4, 8, 5, 1),
    ]
    return MeshData(vertices=vertices, faces=faces)


def shrink_wall_placeholder_size(
    size: Sequence[float],
    xy_scale: float = 0.92,
    z_scale: float = 0.94,
    min_thickness: float = 0.002,
) -> Vector3:
    """Return a slightly smaller wall placeholder size.

    Wall placeholders are only a temporary coarse building body.  Using the
    exact layout bbox creates a solid block that covers generated windows and
    doors in the assembled view.  A small center-preserving shrink keeps the
    wall visible while leaving generated parts proud of the surface.
    """
    sx, sy, sz = (float(value) for value in size)
    return (
        max(sx * float(xy_scale), min_thickness),
        max(sy * float(xy_scale), min_thickness),
        max(sz * float(z_scale), min_thickness),
    )


def read_obj_mesh(path: Union[str, Path]) -> MeshData:
    vertices = []
    faces = []
    with Path(path).open("r") as handle:
        for line in handle:
            if line.startswith("v "):
                _, x, y, z, *unused = line.strip().split()
                vertices.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                face = []
                for token in tokens:
                    index_text = token.split("/")[0]
                    face.append(int(index_text))
                faces.append(tuple(face))
    if not vertices or not faces:
        raise ValueError("OBJ mesh has no vertices or faces: {}".format(path))
    return MeshData(vertices=vertices, faces=faces)


def write_obj_mesh(mesh: MeshData, path: Union[str, Path], header: Optional[str] = None) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        if header:
            for line in header.splitlines():
                handle.write("# {}\n".format(line))
        for vertex in mesh.vertices:
            handle.write("v {:.9f} {:.9f} {:.9f}\n".format(*vertex))
        for face in mesh.faces:
            handle.write("f {}\n".format(" ".join(str(index) for index in face)))
    return str(output_path)


def mesh_bounds(mesh: MeshData) -> Tuple[Vector3, Vector3]:
    xs = [vertex[0] for vertex in mesh.vertices]
    ys = [vertex[1] for vertex in mesh.vertices]
    zs = [vertex[2] for vertex in mesh.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def cleanup_mesh_to_bbox(
    mesh: MeshData,
    center: Sequence[float],
    size: Sequence[float],
    *,
    tolerance_fraction: float = 0.08,
    minimum_component_face_fraction: float = 0.002,
) -> Tuple[MeshData, Dict[str, object]]:
    """Apply generic post-normalization cleanup inside a loose layout bbox.

    This is intentionally model- and class-agnostic.  It removes tiny
    disconnected floaters and drops faces whose vertices are far outside a
    slightly padded target bbox.  The final mesh is renormalized by the caller,
    so this cleanup should reduce obvious pipeline artifacts without enforcing
    semantic class rules.
    """
    target_center = [float(value) for value in center]
    target_size = [float(value) for value in size]
    if not mesh.vertices or not mesh.faces:
        return mesh, {
            "cleanup_enabled": True,
            "removed_vertices": 0,
            "removed_faces": 0,
            "components_before": 0,
            "components_kept": 0,
            "outlier_faces_removed": 0,
        }

    min_allowed = [target_center[i] - target_size[i] * (0.5 + tolerance_fraction) for i in range(3)]
    max_allowed = [target_center[i] + target_size[i] * (0.5 + tolerance_fraction) for i in range(3)]

    inlier_faces: List[Tuple[int, ...]] = []
    outlier_faces_removed = 0
    for face in mesh.faces:
        points = [mesh.vertices[index - 1] for index in face if 1 <= index <= len(mesh.vertices)]
        if len(points) != len(face):
            outlier_faces_removed += 1
            continue
        if all(
            min_allowed[axis] <= point[axis] <= max_allowed[axis]
            for point in points
            for axis in range(3)
        ):
            inlier_faces.append(face)
        else:
            outlier_faces_removed += 1

    if not inlier_faces:
        inlier_faces = list(mesh.faces)
        outlier_faces_removed = 0

    adjacency: Dict[int, set[int]] = {}
    for face_index, face in enumerate(inlier_faces):
        for vertex_index in face:
            adjacency.setdefault(vertex_index, set()).add(face_index)

    face_neighbors: List[set[int]] = [set() for _ in inlier_faces]
    for face_indices in adjacency.values():
        face_list = list(face_indices)
        for left in face_list:
            face_neighbors[left].update(face_list)

    seen = set()
    components: List[List[int]] = []
    for start in range(len(inlier_faces)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in face_neighbors[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    min_component_faces = max(1, int(len(inlier_faces) * minimum_component_face_fraction))
    keep_face_indices = set()
    if components:
        largest_size = max(len(component) for component in components)
        threshold = min(min_component_faces, max(1, int(largest_size * 0.10)))
        for component in components:
            if len(component) >= threshold or len(component) == largest_size:
                keep_face_indices.update(component)
    else:
        keep_face_indices.update(range(len(inlier_faces)))

    kept_faces_original = [inlier_faces[index] for index in sorted(keep_face_indices)]
    used_vertices = sorted({vertex_index for face in kept_faces_original for vertex_index in face})
    remap = {old_index: new_index + 1 for new_index, old_index in enumerate(used_vertices)}
    cleaned_vertices = [mesh.vertices[index - 1] for index in used_vertices]
    cleaned_faces = [tuple(remap[index] for index in face) for face in kept_faces_original]
    cleaned = MeshData(vertices=cleaned_vertices, faces=cleaned_faces)
    return cleaned, {
        "cleanup_enabled": True,
        "removed_vertices": len(mesh.vertices) - len(cleaned_vertices),
        "removed_faces": len(mesh.faces) - len(cleaned_faces),
        "components_before": len(components),
        "components_kept": sum(1 for component in components if any(index in keep_face_indices for index in component)),
        "outlier_faces_removed": outlier_faces_removed,
    }


def normalize_mesh_to_bbox(mesh: MeshData, center: Sequence[float], size: Sequence[float]) -> MeshData:
    source_min, source_max = mesh_bounds(mesh)
    source_size = [source_max[i] - source_min[i] for i in range(3)]
    target_center = [float(value) for value in center]
    target_size = [float(value) for value in size]
    if any(value <= 0 for value in source_size):
        raise ValueError("source mesh has a degenerate axis")

    normalized_vertices = []
    for vertex in mesh.vertices:
        output_vertex = []
        for axis in range(3):
            unit = (vertex[axis] - source_min[axis]) / source_size[axis]
            output_vertex.append(target_center[axis] - target_size[axis] / 2.0 + unit * target_size[axis])
        normalized_vertices.append(tuple(output_vertex))
    return MeshData(vertices=normalized_vertices, faces=list(mesh.faces))


def hunyuan_y_up_to_layout_z_up(mesh: MeshData) -> MeshData:
    """Convert Hunyuan/model-viewer y-up mesh axes into BuildingBlock z-up axes.

    Hunyuan3D/GLB-style meshes use the second coordinate as vertical. The
    BuildingBlock layout contract uses z as vertical.  If we normalize raw
    Hunyuan vertices axis-by-axis without this conversion, tall doors/windows
    get squeezed into layout Y/depth and appear laid over or inverted in the
    assembled building.
    """
    converted_vertices = []
    for x, y, z in mesh.vertices:
        converted_vertices.append((float(x), -float(z), float(y)))
    return MeshData(vertices=converted_vertices, faces=list(mesh.faces))


def normalize_mesh_file_to_bbox(
    source_path: Union[str, Path],
    output_path: Union[str, Path],
    contract_part: Mapping[str, object],
    *,
    cleanup: bool = False,
) -> str:
    if Path(source_path).suffix.lower() != ".obj":
        raise ValueError("minimal V1 normalization supports OBJ meshes only: {}".format(source_path))
    center, size = bbox_center_size(contract_part)
    mesh = hunyuan_y_up_to_layout_z_up(read_obj_mesh(source_path))
    normalized = normalize_mesh_to_bbox(mesh, center, size)
    cleanup_info = None
    if cleanup:
        cleaned, cleanup_info = cleanup_mesh_to_bbox(normalized, center, size)
        normalized = normalize_mesh_to_bbox(cleaned, center, size)
        info_path = Path(output_path).with_suffix(".cleanup.json")
        info_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        info_path.write_text(json.dumps(cleanup_info, indent=2, sort_keys=True) + "\n")
    return write_obj_mesh(
        normalized,
        output_path,
        header="normalized part_id={} cleanup={}".format(contract_part.get("part_id"), bool(cleanup_info)),
    )


def write_placeholder_mesh(
    contract_part: Mapping[str, object],
    output_path: Union[str, Path],
    *,
    shrink_wall: bool = False,
    wall_xy_scale: float = 0.92,
    wall_z_scale: float = 0.94,
) -> str:
    center, size = bbox_center_size(contract_part)
    if shrink_wall and str(contract_part.get("target_class", "")).startswith("wall"):
        size = shrink_wall_placeholder_size(
            size,
            xy_scale=wall_xy_scale,
            z_scale=wall_z_scale,
        )
    mesh = create_box_mesh(center, size)
    return write_obj_mesh(
        mesh,
        output_path,
        header="placeholder part_id={} shrink_wall={}".format(
            contract_part.get("part_id"),
            bool(shrink_wall),
        ),
    )


def _result_value(result: object, key: str, default=None):
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def build_assemblies(
    contract_parts: Iterable[Union[str, Path, Mapping[str, object]]],
    hunyuan_results: Iterable[object],
    output_dir: Union[str, Path],
) -> AssemblyResult:
    contracts = [load_contract_part(part) for part in contract_parts]
    results_by_part_id = {
        str(_result_value(result, "part_id")): result for result in hunyuan_results
    }
    output_path = Path(output_dir)
    parts_dir = output_path / "parts"
    placeholders_dir = output_path / "placeholders"
    raw_mesh = MeshData(vertices=[], faces=[])
    placeholder_mesh = MeshData(vertices=[], faces=[])
    parts = []
    layout_id = str(contracts[0].get("layout_id", "")) if contracts else ""
    schema_version = str(contracts[0].get("schema_version", "")) if contracts else ""

    for contract in contracts:
        part_id = str(contract["part_id"])
        result = results_by_part_id.get(part_id, {})
        source_raw_path = _result_value(result, "raw_output_path")
        lifecycle_states = list(_result_value(result, "lifecycle_states", []))
        raw_hunyuan_output_path = source_raw_path
        normalized_output_path = None

        if source_raw_path:
            normalized_path = parts_dir / "{}.obj".format(part_id)
            try:
                normalized_output_path = normalize_mesh_file_to_bbox(source_raw_path, normalized_path, contract)
                raw_mesh.extend(read_obj_mesh(normalized_output_path))
                lifecycle_states.append("assembled_raw")
            except Exception:
                source_raw_path = None
                normalized_output_path = None
                lifecycle_states = [
                    state
                    for state in lifecycle_states
                    if state not in ("hunyuan_succeeded", "assembled_raw")
                ]
                lifecycle_states.extend([FAILURE_NORMALIZATION, "hunyuan_failed"])
        else:
            if "hunyuan_failed" not in lifecycle_states:
                lifecycle_states.append("hunyuan_failed")

        placeholder_output_path = normalized_output_path
        if normalized_output_path:
            placeholder_mesh.extend(read_obj_mesh(normalized_output_path))
            lifecycle_states.append("assembled_placeholder")
        else:
            if "submitted_to_hunyuan" not in lifecycle_states:
                raise ValueError(
                    "placeholder fallback requires a submitted Hunyuan attempt for part {}".format(
                        part_id
                    )
                )
            lifecycle_states.append("placeholder_used")
            placeholder_output_path = write_placeholder_mesh(
                contract,
                placeholders_dir / "{}__placeholder.obj".format(part_id),
            )
            placeholder_mesh.extend(read_obj_mesh(placeholder_output_path))
            lifecycle_states.append("assembled_placeholder")

        parts.append(
            AssemblyPartResult(
                part_id=part_id,
                raw_hunyuan_output_path=raw_hunyuan_output_path,
                normalized_output_path=normalized_output_path,
                placeholder_output_path=placeholder_output_path,
                lifecycle_states=lifecycle_states,
            )
        )

    raw_assembly_path = None
    if raw_mesh.vertices:
        raw_assembly_path = write_obj_mesh(
            raw_mesh,
            output_path / "raw_hunyuan_assembly.obj",
            header="raw_hunyuan_assembly layout_id={} schema_version={}".format(
                layout_id, schema_version
            ),
        )

    placeholder_assembly_path = write_obj_mesh(
        placeholder_mesh,
        output_path / "placeholder_filled_assembly.obj",
        header="placeholder_filled_assembly layout_id={} schema_version={}".format(
            layout_id, schema_version
        ),
    )

    return AssemblyResult(
        layout_id=layout_id,
        schema_version=schema_version,
        raw_assembly_path=raw_assembly_path,
        placeholder_assembly_path=placeholder_assembly_path,
        parts=parts,
    )
