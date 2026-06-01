"""Lightweight visualization helpers for BuildingBlock-Hunyuan reports."""

import math
import json
import os
import struct
from pathlib import Path
import random
import textwrap


CLASS_COLORS = {
    "wall": "#4C78A8",
    "window": "#54A24B",
    "door": "#F58518",
    "roof": "#E45756",
    "balcony": "#B279A2",
    "chimney": "#9D755D",
    "object": "#BAB0AC",
}

PART_COLOR_SEQUENCE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
    "#FF9DA6", "#9D755D", "#BAB0AC", "#ECA82C", "#5F4690", "#1D6996",
    "#38A6A5", "#0F8554", "#73AF48", "#EDAD08", "#E17C05", "#CC503E",
    "#94346E", "#6F4070", "#994F88", "#1F77B4", "#2CA02C", "#D62728",
]


GLB_VIEWER_COORDINATE_FRAME = {
    "source": "BuildingBlock layout frame: right-handed x=right, y=forward, z=up",
    "target": "glTF/model-viewer frame: right-handed x=right, y=up, z=back",
    "vertex_map": "(x, y, z) -> (x, z, -y)",
}


def part_color(part, index=0):
    """Deterministic per-part color for overall assembly/layout correspondence."""
    part_id = str(part.get("part_id", ""))
    if part_id:
        total = sum(ord(ch) for ch in part_id)
    else:
        total = int(index)
    return PART_COLOR_SEQUENCE[total % len(PART_COLOR_SEQUENCE)]


def layout_vertex_to_glb_viewer(vertex):
    """Map BuildingBlock z-up layout coordinates into glTF's y-up viewer frame."""
    x, y, z = (float(value) for value in vertex)
    return (x, z, -y)


def _transform_vertices_for_glb(vertices):
    return [layout_vertex_to_glb_viewer(vertex) for vertex in vertices]


def _hex_to_rgba255(color, alpha=255):
    color = color.lstrip("#")
    return [
        int(color[0:2], 16),
        int(color[2:4], 16),
        int(color[4:6], 16),
        int(alpha),
    ]


def _align4_bytes(payload):
    padding = (-len(payload)) % 4
    if padding:
        payload += b"\x00" * padding
    return payload


def _align4_json_text(text):
    payload = text.encode("utf-8")
    padding = (-len(payload)) % 4
    if padding:
        payload += b" " * padding
    return payload


def _box_vertices_and_triangles(center, size):
    cx, cy, cz = (float(v) for v in center)
    sx, sy, sz = (float(v) for v in size)
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
    triangles = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    return vertices, triangles


def _read_obj_full_mesh(path):
    vertices = []
    triangles = []
    with Path(path).open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                face = _parse_obj_face_indices(line.split()[1:], len(vertices))
                if not face:
                    continue
                for index in range(1, len(face) - 1):
                    triangles.append((face[0], face[index], face[index + 1]))
    return vertices, triangles


def _compute_vertex_normals(vertices, triangles):
    normals = [[0.0, 0.0, 0.0] for _ in vertices]
    for i0, i1, i2 in triangles:
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
        edge1 = _sub(v1, v0)
        edge2 = _sub(v2, v0)
        face_normal = _cross(edge1, edge2)
        for index in (i0, i1, i2):
            normals[index][0] += face_normal[0]
            normals[index][1] += face_normal[1]
            normals[index][2] += face_normal[2]
    return [_normalize(tuple(normal)) for normal in normals]


def _compact_indexed_mesh(vertices, triangles):
    if not vertices or not triangles:
        return [], []
    used_indices = []
    seen = set()
    for triangle in triangles:
        for index in triangle:
            if index not in seen:
                seen.add(index)
                used_indices.append(index)
    remap = {old_index: new_index for new_index, old_index in enumerate(used_indices)}
    compact_vertices = [vertices[index] for index in used_indices]
    compact_triangles = [
        (remap[i0], remap[i1], remap[i2])
        for i0, i1, i2 in triangles
    ]
    return compact_vertices, compact_triangles


def _vertex_cell_key(vertex, cell_size):
    return tuple(int(round(float(value) / cell_size)) for value in vertex)


def _weld_vertices_by_grid(vertices, triangles, max_vertices=None):
    """Reduce dense meshes by welding nearby vertices while preserving faces.

    This is intentionally different from stride/random face sampling: every
    remaining triangle still belongs to a connected surface patch, so the web
    viewer reads as a mesh rather than a point/triangle cloud.
    """
    if not max_vertices or len(vertices) <= max_vertices:
        return vertices, triangles

    bounds_min, bounds_max = _mesh_bounds(vertices)
    diag = math.sqrt(sum((bounds_max[i] - bounds_min[i]) ** 2 for i in range(3)))
    if diag <= 0:
        return vertices, triangles

    cell_size = diag / 1000.0
    best_vertices = vertices
    best_triangles = triangles
    for _ in range(18):
        remap = {}
        welded_vertices = []
        source_to_welded = []
        for vertex in vertices:
            key = _vertex_cell_key(vertex, cell_size)
            index = remap.get(key)
            if index is None:
                index = len(welded_vertices)
                remap[key] = index
                welded_vertices.append(vertex)
            source_to_welded.append(index)

        welded_triangles = []
        seen_triangles = set()
        for i0, i1, i2 in triangles:
            tri = (
                source_to_welded[i0],
                source_to_welded[i1],
                source_to_welded[i2],
            )
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                continue
            key = tuple(sorted(tri))
            if key in seen_triangles:
                continue
            seen_triangles.add(key)
            welded_triangles.append(tri)

        if welded_vertices and welded_triangles:
            best_vertices, best_triangles = welded_vertices, welded_triangles
        if len(welded_vertices) <= max_vertices:
            return welded_vertices, welded_triangles
        cell_size *= 1.45

    return best_vertices, best_triangles


def _pack_vec3_f32(values):
    payload = bytearray()
    for x, y, z in values:
        payload.extend(struct.pack("<3f", float(x), float(y), float(z)))
    return bytes(payload)


def _pack_scalar_u32(values):
    payload = bytearray()
    for value in values:
        payload.extend(struct.pack("<I", int(value)))
    return bytes(payload)


def _mesh_bounds(vertices):
    xs = [vertex[0] for vertex in vertices]
    ys = [vertex[1] for vertex in vertices]
    zs = [vertex[2] for vertex in vertices]
    return [min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]


def _write_glb(primitives, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not primitives:
        return None

    materials = []
    meshes = []
    nodes = []
    accessors = []
    buffer_views = []
    binary_chunks = []
    byte_offset = 0

    def append_buffer_view(payload, target=None):
        nonlocal byte_offset
        payload = _align4_bytes(payload)
        offset = byte_offset
        binary_chunks.append(payload)
        byte_offset += len(payload)
        buffer_view = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        }
        if target is not None:
            buffer_view["target"] = target
        buffer_views.append(buffer_view)
        return len(buffer_views) - 1

    def append_accessor(buffer_view_index, component_type, count, accessor_type, *, mins=None, maxs=None):
        accessor = {
            "bufferView": buffer_view_index,
            "componentType": component_type,
            "count": count,
            "type": accessor_type,
        }
        if mins is not None:
            accessor["min"] = mins
        if maxs is not None:
            accessor["max"] = maxs
        accessors.append(accessor)
        return len(accessors) - 1

    for primitive in primitives:
        vertices = primitive["vertices"]
        triangles = primitive["triangles"]
        if not vertices or not triangles:
            continue
        normals = primitive.get("normals") or _compute_vertex_normals(vertices, triangles)
        flat_indices = [index for tri in triangles for index in tri]
        bounds_min, bounds_max = _mesh_bounds(vertices)
        position_view = append_buffer_view(_pack_vec3_f32(vertices), target=34962)
        normal_view = append_buffer_view(_pack_vec3_f32(normals), target=34962)
        index_view = append_buffer_view(_pack_scalar_u32(flat_indices), target=34963)
        position_accessor = append_accessor(position_view, 5126, len(vertices), "VEC3", mins=bounds_min, maxs=bounds_max)
        normal_accessor = append_accessor(normal_view, 5126, len(normals), "VEC3")
        index_accessor = append_accessor(index_view, 5125, len(flat_indices), "SCALAR")
        material = {
            "pbrMetallicRoughness": {
                "baseColorFactor": [
                    primitive["rgba"][0] / 255.0,
                    primitive["rgba"][1] / 255.0,
                    primitive["rgba"][2] / 255.0,
                    primitive["rgba"][3] / 255.0,
                ],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "doubleSided": True,
        }
        if primitive["rgba"][3] < 255:
            material["alphaMode"] = "BLEND"
        materials.append(material)
        meshes.append({
            "primitives": [{
                "attributes": {
                    "POSITION": position_accessor,
                    "NORMAL": normal_accessor,
                },
                "indices": index_accessor,
                "material": len(materials) - 1,
            }]
        })
        nodes.append({"mesh": len(meshes) - 1, "name": primitive.get("name", "mesh")})

    if not meshes:
        return None

    binary_blob = b"".join(binary_chunks)
    gltf = {
        "asset": {"version": "2.0", "generator": "BuildingBlock minimal glb exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "buffers": [{"byteLength": len(binary_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }
    json_chunk = _align4_json_text(json.dumps(gltf, separators=(",", ":")))
    bin_chunk = _align4_bytes(binary_blob)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with output_path.open("wb") as handle:
        handle.write(struct.pack("<4sII", b"glTF", 2, total_length))
        handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        handle.write(json_chunk)
        handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\x00"))
        handle.write(bin_chunk)
    return str(output_path)


def export_obj_mesh_glb(
    obj_path,
    output_path,
    color="#54A24B",
    name=None,
    target_faces=None,
):
    """Export one OBJ mesh as a colored, y-up GLB for model-viewer."""
    vertices, triangles = _read_obj_full_mesh(obj_path)
    if not vertices or not triangles:
        return None
    if target_faces and len(triangles) > target_faces:
        stride = max(1, len(triangles) // target_faces)
        triangles = triangles[::stride][:target_faces]
    vertices, triangles = _compact_indexed_mesh(vertices, triangles)
    if not vertices or not triangles:
        return None
    primitive = {
        "name": name or Path(obj_path).stem,
        "vertices": _transform_vertices_for_glb(vertices),
        "triangles": triangles,
        "rgba": _hex_to_rgba255(color, alpha=255),
    }
    return _write_glb([primitive], output_path)


def _set_equal_3d(ax, xs, ys, zs):
    ranges = [max(values) - min(values) for values in (xs, ys, zs)]
    centers = [(max(values) + min(values)) / 2 for values in (xs, ys, zs)]
    radius = max(ranges) / 2 or 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _set_equal_3d_bounds(ax, bounds_min, bounds_max):
    xs = (bounds_min[0], bounds_max[0])
    ys = (bounds_min[1], bounds_max[1])
    zs = (bounds_min[2], bounds_max[2])
    _set_equal_3d(ax, xs, ys, zs)


def read_obj_vertices(path, max_vertices=50000):
    vertices = []
    rng = random.Random(1234)
    seen = 0
    with Path(path).open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
                    seen += 1
                    if len(vertices) < max_vertices:
                        vertices.append(vertex)
                    else:
                        replacement_index = rng.randrange(seen)
                        if replacement_index < max_vertices:
                            vertices[replacement_index] = vertex
    return vertices


def _parse_obj_face_indices(tokens, vertex_count):
    indices = []
    for token in tokens:
        index_text = token.split("/")[0]
        if not index_text:
            continue
        index = int(index_text)
        if index < 0:
            index = vertex_count + index + 1
        if index > 0:
            indices.append(index - 1)
    return indices if len(indices) >= 3 else None


def _parse_obj_face_vertex_uv(tokens, vertex_count, uv_count):
    vertices = []
    uv_indices = []
    for token in tokens:
        chunks = token.split("/")
        if not chunks or not chunks[0]:
            continue
        vertex_index = int(chunks[0])
        if vertex_index < 0:
            vertex_index = vertex_count + vertex_index + 1
        if vertex_index <= 0:
            continue
        vertices.append(vertex_index - 1)
        uv_index = None
        if len(chunks) > 1 and chunks[1]:
            parsed_uv = int(chunks[1])
            if parsed_uv < 0:
                parsed_uv = uv_count + parsed_uv + 1
            if parsed_uv > 0:
                uv_index = parsed_uv - 1
        uv_indices.append(uv_index)
    if len(vertices) < 3:
        return None
    return {"vertices": vertices, "uvs": uv_indices}


def _parse_obj_mtl_libraries(obj_path):
    libraries = []
    obj_path = Path(obj_path)
    try:
        with obj_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("mtllib "):
                    for token in line.strip().split()[1:]:
                        candidate = obj_path.parent / token
                        if candidate.exists():
                            libraries.append(candidate)
    except Exception:
        return []
    return libraries


def _parse_mtl_diffuse_maps(obj_path):
    material_to_texture = {}
    for mtl_path in _parse_obj_mtl_libraries(obj_path):
        current = None
        for raw in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("newmtl "):
                current = line.split(maxsplit=1)[1].strip()
                continue
            if current and line.startswith("map_Kd "):
                # MTL map_Kd may contain options; the last token is the filename
                # for the simple OBJ/MTL assets produced by this project.
                texture_name = line.split()[-1]
                texture_path = mtl_path.parent / texture_name
                if texture_path.exists():
                    material_to_texture[current] = texture_path
    return material_to_texture


def _load_texture_array(texture_path, cache):
    key = str(texture_path)
    if key in cache:
        return cache[key]
    try:
        from PIL import Image
        image = Image.open(texture_path).convert("RGB")
        # Downsample very large atlases for evidence rendering.  The QA image is
        # only a color/material witness; exact texel fidelity is unnecessary.
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
        array = __import__("numpy").asarray(image, dtype="float32") / 255.0
        cache[key] = array
        return array
    except Exception:
        cache[key] = None
        return None


def _sample_texture_color(texture_path, uv_coords, cache):
    array = _load_texture_array(texture_path, cache)
    if array is None or array.size == 0:
        return None
    if not uv_coords:
        rgb = array.reshape(-1, 3).mean(axis=0)
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    u = sum(float(item[0]) for item in uv_coords) / len(uv_coords)
    v = sum(float(item[1]) for item in uv_coords) / len(uv_coords)
    # OBJ UVs wrap by convention; image origin is top-left while OBJ v=0 is
    # bottom, so flip v for a visually plausible material witness.
    u = u % 1.0
    v = v % 1.0
    height, width = array.shape[:2]
    x = min(width - 1, max(0, int(round(u * (width - 1)))))
    y = min(height - 1, max(0, int(round((1.0 - v) * (height - 1)))))
    rgb = array[y, x, :3]
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def read_obj_mesh_sample(path, max_faces=12000, use_material_colors=False):
    if not use_material_colors and str(os.environ.get("S2_RENDER_TRIMESH_SIMPLIFY", "1")).lower() not in {"0", "false", "no", "off"}:
        try:
            import trimesh
            mesh = trimesh.load_mesh(str(path), process=False)
            if isinstance(mesh, trimesh.Scene):
                meshes = [geom for geom in mesh.geometry.values() if hasattr(geom, "vertices") and hasattr(geom, "faces")]
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
            if hasattr(mesh, "vertices") and hasattr(mesh, "faces") and len(mesh.vertices) and len(mesh.faces):
                target_faces = int(max_faces or 0)
                if target_faces > 0 and len(mesh.faces) > target_faces:
                    try:
                        mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
                    except Exception:
                        stride = max(1, int(math.ceil(len(mesh.faces) / float(target_faces))))
                        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[::stride][:target_faces], process=False)
                vertices_np = mesh.vertices
                faces_np = mesh.faces
                polygons = [
                    [tuple(float(v) for v in vertices_np[int(index)]) for index in face]
                    for face in faces_np
                    if len(face) >= 3
                ]
                if polygons:
                    bounds = mesh.bounds
                    return {
                        "polygons": polygons,
                        "bounds_min": tuple(float(value) for value in bounds[0]),
                        "bounds_max": tuple(float(value) for value in bounds[1]),
                        "surface_decimation": "trimesh_quadric",
                        "source_face_count": int(len(faces_np)),
                    }
        except Exception:
            pass

    vertices = []
    uvs = []
    sampled_faces = []
    rng = random.Random(5678)
    faces_seen = 0
    current_material = None
    material_to_texture = _parse_mtl_diffuse_maps(path) if use_material_colors else {}
    texture_cache = {}
    bounds_min = [float("inf"), float("inf"), float("inf")]
    bounds_max = [float("-inf"), float("-inf"), float("-inf")]

    with Path(path).open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertex = (float(parts[1]), float(parts[2]), float(parts[3]))
                    vertices.append(vertex)
                    for axis, value in enumerate(vertex):
                        bounds_min[axis] = min(bounds_min[axis], value)
                        bounds_max[axis] = max(bounds_max[axis], value)
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        uvs.append((float(parts[1]), float(parts[2])))
                    except ValueError:
                        pass
            elif line.startswith("usemtl "):
                current_material = line.split(maxsplit=1)[1].strip()
            elif line.startswith("f "):
                if use_material_colors:
                    face = _parse_obj_face_vertex_uv(line.split()[1:], len(vertices), len(uvs))
                else:
                    vertices_only = _parse_obj_face_indices(line.split()[1:], len(vertices))
                    face = {"vertices": vertices_only, "uvs": []} if vertices_only is not None else None
                if face is None:
                    continue
                face["material"] = current_material
                faces_seen += 1
                if len(sampled_faces) < max_faces:
                    sampled_faces.append(face)
                else:
                    replacement_index = rng.randrange(faces_seen)
                    if replacement_index < max_faces:
                        sampled_faces[replacement_index] = face

    polygons = []
    face_colors = []
    for face in sampled_faces:
        try:
            polygons.append([vertices[index] for index in face["vertices"]])
        except (IndexError, KeyError):
            continue
        if use_material_colors:
            texture_path = material_to_texture.get(face.get("material"))
            color = None
            if texture_path:
                uv_coords = []
                for uv_index in face.get("uvs") or []:
                    if uv_index is not None and 0 <= uv_index < len(uvs):
                        uv_coords.append(uvs[uv_index])
                color = _sample_texture_color(texture_path, uv_coords, texture_cache)
            face_colors.append(color)

    if not vertices:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]

    result = {
        "polygons": polygons,
        "bounds_min": tuple(bounds_min),
        "bounds_max": tuple(bounds_max),
    }
    if use_material_colors:
        result["face_base_colors"] = face_colors
    return result


DEFAULT_3D_VIEWS = {
    "iso": (24, 42),
    "front": (8, 0),
    "side": (8, 90),
    "top": (90, -90),
}


def render_obj_points(
    obj_path,
    output_path,
    title=None,
    color="#4C78A8",
    max_vertices=50000,
    elev=24,
    azim=42,
    figsize=(5, 4),
    point_size=0.7,
):
    vertices = read_obj_vertices(obj_path, max_vertices=max_vertices)
    return render_point_cloud(
        vertices,
        output_path,
        title=title,
        color=color,
        elev=elev,
        azim=azim,
        figsize=figsize,
        point_size=point_size,
    )


def render_point_cloud(
    vertices,
    output_path,
    title=None,
    color="#4C78A8",
    elev=24,
    azim=42,
    figsize=(5, 4),
    point_size=0.7,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not vertices:
        return None
    xs, ys, zs = zip(*vertices)
    stride = max(1, len(vertices) // 12000)
    xs, ys, zs = xs[::stride], ys[::stride], zs[::stride]
    fig = plt.figure(figsize=figsize, dpi=150, facecolor="#101318")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#101318")
    ax.scatter(xs, ys, zs, s=point_size, c=color, alpha=0.95, depthshade=True)
    _set_equal_3d(ax, xs, ys, zs)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, color="#172033")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(output_path)


def render_obj_point_views(obj_path, output_dir, prefix, title="Mesh"):
    outputs = {}
    vertices = read_obj_vertices(obj_path, max_vertices=90000)
    for view_name, (elev, azim) in DEFAULT_3D_VIEWS.items():
        rendered = render_point_cloud(
            vertices,
            Path(output_dir) / f"{prefix}_{view_name}.png",
            title=f"{title} - {view_name}",
            color="#54A24B",
            elev=elev,
            azim=azim,
            figsize=(6, 4.8),
            point_size=0.55,
        )
        if rendered:
            outputs[view_name] = rendered
    return outputs


def render_mesh_polygons(
    mesh_sample,
    output_path,
    title=None,
    color="#54A24B",
    elev=24,
    azim=42,
    figsize=(6, 4.8),
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    polygons = mesh_sample["polygons"]
    if not polygons:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    projected, face_colors, bounds_2d = _project_mesh_polygons(
        polygons,
        mesh_sample["bounds_min"],
        mesh_sample["bounds_max"],
        elev=elev,
        azim=azim,
        color=color,
        face_base_colors=mesh_sample.get("face_base_colors"),
    )
    if not projected:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor="#f7fafc")
    ax.set_facecolor("#f7fafc")
    collection = PolyCollection(
        projected,
        facecolors=face_colors,
        edgecolors=(0.18, 0.22, 0.28, 0.0),
        linewidths=0.0,
        closed=True,
    )
    ax.add_collection(collection)
    min_x, max_x, min_y, max_y = bounds_2d
    pad = max(max_x - min_x, max_y - min_y) * 0.04 or 0.1
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, color="#172033")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(output_path)


def _hex_to_rgb01(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _normalize(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        return (0.0, 0.0, 0.0)
    return tuple(value / length for value in vector)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _view_basis(elev, azim):
    elev_rad = math.radians(elev)
    azim_rad = math.radians(azim)
    view_dir = _normalize((
        math.cos(elev_rad) * math.cos(azim_rad),
        math.cos(elev_rad) * math.sin(azim_rad),
        math.sin(elev_rad),
    ))
    right = _normalize((-math.sin(azim_rad), math.cos(azim_rad), 0.0))
    up = _normalize(_cross(right, view_dir))
    return right, up, view_dir


def _project_point(point, right, up, view_dir):
    return (_dot(point, right), _dot(point, up), _dot(point, view_dir))


def _bounds_corners(bounds_min, bounds_max):
    return [
        (x, y, z)
        for x in (bounds_min[0], bounds_max[0])
        for y in (bounds_min[1], bounds_max[1])
        for z in (bounds_min[2], bounds_max[2])
    ]


def _brighten_rgb(rgb, mix=0.28):
    return tuple(min(1.0, float(channel) * (1.0 - mix) + mix) for channel in rgb)


def _project_mesh_polygons(polygons, bounds_min, bounds_max, elev, azim, color, face_base_colors=None):
    right, up, view_dir = _view_basis(elev, azim)
    fallback_rgb = _brighten_rgb(_hex_to_rgb01(color), mix=0.30)
    light_dir = _normalize((0.35, -0.45, 0.82))
    view_light_dir = _normalize((view_dir[0] + 0.25, view_dir[1] - 0.20, view_dir[2] + 0.65))
    projected_with_depth = []

    for polygon_index, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        projected = [_project_point(point, right, up, view_dir) for point in polygon]
        points_2d = [(point[0], point[1]) for point in projected]
        depth = sum(point[2] for point in projected) / len(projected)
        normal = _normalize(_cross(_sub(polygon[1], polygon[0]), _sub(polygon[2], polygon[0])))
        # High ambient, low-contrast shading makes dense Hunyuan triangle meshes
        # read as continuous bright surfaces instead of dark point/noise fields.
        diffuse = max(abs(_dot(normal, light_dir)), abs(_dot(normal, view_light_dir)))
        shade = 0.72 + 0.28 * diffuse
        base_rgb = fallback_rgb
        if face_base_colors and polygon_index < len(face_base_colors) and face_base_colors[polygon_index] is not None:
            base_rgb = _brighten_rgb(face_base_colors[polygon_index], mix=0.18)
        face_color = (
            min(float(base_rgb[0]) * shade, 1.0),
            min(float(base_rgb[1]) * shade, 1.0),
            min(float(base_rgb[2]) * shade, 1.0),
            0.99,
        )
        projected_with_depth.append((depth, points_2d, face_color))

    projected_with_depth.sort(key=lambda item: item[0])
    projected = [item[1] for item in projected_with_depth]
    face_colors = [item[2] for item in projected_with_depth]

    projected_bounds = [
        _project_point(corner, right, up, view_dir)
        for corner in _bounds_corners(bounds_min, bounds_max)
    ]
    xs = [point[0] for point in projected_bounds]
    ys = [point[1] for point in projected_bounds]
    bounds_2d = (min(xs), max(xs), min(ys), max(ys))
    return projected, face_colors, bounds_2d


def _transform_mesh_sample(mesh_sample, transform):
    if transform is None:
        return mesh_sample
    polygons = [
        [transform(point) for point in polygon]
        for polygon in mesh_sample["polygons"]
    ]
    transformed_bounds = [transform(point) for point in _bounds_corners(
        mesh_sample["bounds_min"],
        mesh_sample["bounds_max"],
    )]
    bounds_min = tuple(min(point[axis] for point in transformed_bounds) for axis in range(3))
    bounds_max = tuple(max(point[axis] for point in transformed_bounds) for axis in range(3))
    return {
        **mesh_sample,
        "polygons": polygons,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
    }


def render_obj_mesh(obj_path, output_path, title=None, color="#54A24B", max_faces=12000, elev=24, azim=42, figsize=(6, 4.8)):
    mesh_sample = read_obj_mesh_sample(obj_path, max_faces=max_faces)
    rendered = render_mesh_polygons(
        mesh_sample,
        output_path,
        title=title,
        color=color,
        elev=elev,
        azim=azim,
        figsize=figsize,
    )
    if rendered:
        return rendered
    return render_obj_points(
        obj_path,
        output_path,
        title=title,
        color=color,
        max_vertices=50000,
        elev=elev,
        azim=azim,
        figsize=figsize,
    )


def render_obj_mesh_strict(
    obj_path,
    output_path,
    title=None,
    color="#54A24B",
    max_faces=22000,
    elev=24,
    azim=42,
    figsize=(6, 4.8),
):
    """Render only mesh surfaces; return None instead of falling back to points."""
    mesh_sample = read_obj_mesh_sample(obj_path, max_faces=max_faces)
    return render_mesh_polygons(
        mesh_sample,
        output_path,
        title=title,
        color=color,
        elev=elev,
        azim=azim,
        figsize=figsize,
    )


def render_obj_mesh_views(obj_path, output_dir, prefix, title="Mesh", max_faces=12000, use_material_colors=False, views=None):
    mesh_sample = read_obj_mesh_sample(obj_path, max_faces=max_faces, use_material_colors=use_material_colors)
    outputs = {}
    view_map = views or DEFAULT_3D_VIEWS
    for view_name, (elev, azim) in view_map.items():
        rendered = render_mesh_polygons(
            mesh_sample,
            Path(output_dir) / f"{prefix}_{view_name}.png",
            title=f"{title} - {view_name}",
            color="#54A24B",
            elev=elev,
            azim=azim,
            figsize=(6, 4.8),
        )
        if rendered:
            outputs[view_name] = rendered
    # VLM evidence must not fall back to vertex/point-cloud renders.  A dense
    # Hunyuan mesh rendered as sampled points is visually misleading and makes
    # solid geometry look like point cloud.  If surface rendering fails, return
    # no mesh view so callers can use other evidence instead of sending a bad
    # image to the VLM.
    return outputs



S1_STYLE_LAYOUT_VLM_VIEWS = {
    "iso": (28, -42),
    "left_oblique": (16, -70),
    "right_oblique": (16, 35),
}

S2_VLM_SIX_VIEWS = {
    "iso": (24, 42),
    "front": (8, 0),
    "back": (8, 180),
    "left": (8, -90),
    "right": (8, 90),
    "top": (90, -90),
}


def _layout_part_records(contract):
    records = []
    for index, part in enumerate(contract.get("parts", []) or []):
        bbox = part.get("bbox") if isinstance(part.get("bbox"), dict) else {}
        center = bbox.get("center", part.get("actor_location", part.get("center")))
        size = bbox.get("size", part.get("actor_size", part.get("size")))
        if not isinstance(center, (list, tuple)) or not isinstance(size, (list, tuple)) or len(center) < 3 or len(size) < 3:
            continue
        records.append({
            "part": part,
            "part_id": str(part.get("part_id") or part.get("actor_label") or f"part_{index + 1:02d}"),
            "index": index,
            "number": index + 1,
            "label": (
                part.get("part_description")
                or part.get("open_vocab_label")
                or part.get("source_actor_label")
                or part.get("target_class")
                or f"part_{index + 1:02d}"
            ),
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "size": [max(abs(float(size[0])), 1e-4), max(abs(float(size[1])), 1e-4), max(abs(float(size[2])), 1e-4)],
        })
    return records


def _bbox_faces_from_vertices(vertices):
    return [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [3, 0, 4, 7]],
    ]


def _set_equal_3d_for_records(ax, records):
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for record in records:
        center = record["center"]
        size = record["size"]
        for axis in range(3):
            mins[axis] = min(mins[axis], center[axis] - size[axis] / 2.0)
            maxs[axis] = max(maxs[axis], center[axis] + size[axis] / 2.0)
    if any(value in (float("inf"), float("-inf")) for value in mins + maxs):
        mins, maxs = [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
    spans = [maxs[i] - mins[i] for i in range(3)]
    radius = max(max(spans) / 2.0, 0.5) * 1.18
    centers = [(mins[i] + maxs[i]) / 2.0 for i in range(3)]
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _axes_overlay(ax):
    fig = ax.get_figure()
    overlay = fig.add_axes(ax.get_position(), frameon=False, zorder=10000)
    overlay.set_axes_locator(lambda _overlay, _renderer, target=ax: target.get_position())
    overlay.set_xlim(0.0, 1.0)
    overlay.set_ylim(0.0, 1.0)
    overlay.set_axis_off()
    overlay.patch.set_alpha(0.0)
    overlay.set_in_layout(False)
    return overlay


def _project_point_axes(ax, point):
    from mpl_toolkits.mplot3d import proj3d

    x2, y2, depth = proj3d.proj_transform(point[0], point[1], point[2], ax.get_proj())
    x_fig, y_fig = ax.transData.transform((x2, y2))
    x_axes, y_axes = ax.transAxes.inverted().transform((x_fig, y_fig))
    return float(x_axes), float(y_axes), float(depth)


def _point_in_polygon_2d(point, polygon):
    x, y = point
    inside = False
    count = len(polygon)
    if count < 3:
        return False
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _subsample_points(points, limit=90):
    if len(points) <= limit:
        return points
    step = max(1, len(points) // limit)
    return points[::step][:limit]


def _pixel_visible_items_from_polygons(polygon_items, resolution=96, min_visible_pixels=5):
    if not polygon_items:
        return []
    resolution = max(32, int(resolution))
    owner = [[None for _ in range(resolution)] for _ in range(resolution)]
    depth_grid = [[float("-inf") for _ in range(resolution)] for _ in range(resolution)]
    for item in polygon_items:
        polygon = [(float(x), float(y)) for x, y in item.get("polygon", [])]
        if len(polygon) < 3:
            continue
        xs = [x for x, _ in polygon]
        ys = [y for _, y in polygon]
        ix0 = max(0, int(math.floor(min(xs) * resolution)))
        ix1 = min(resolution - 1, int(math.ceil(max(xs) * resolution)))
        iy0 = max(0, int(math.floor(min(ys) * resolution)))
        iy1 = min(resolution - 1, int(math.ceil(max(ys) * resolution)))
        if ix1 < ix0 or iy1 < iy0:
            continue
        depth = float(item.get("depth", 0.0))
        for iy in range(iy0, iy1 + 1):
            y = (iy + 0.5) / resolution
            for ix in range(ix0, ix1 + 1):
                x = (ix + 0.5) / resolution
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    continue
                if not _point_in_polygon_2d((x, y), polygon):
                    continue
                if depth > depth_grid[iy][ix] + 1e-9:
                    depth_grid[iy][ix] = depth
                    owner[iy][ix] = item
    points_by_number = {}
    item_by_number = {}
    for iy in range(resolution):
        for ix in range(resolution):
            item = owner[iy][ix]
            if not item:
                continue
            number = int(item["record"]["number"])
            points_by_number.setdefault(number, []).append(((ix + 0.5) / resolution, (iy + 0.5) / resolution))
            item_by_number[number] = item
    visible = []
    for number, points in points_by_number.items():
        if len(points) < min_visible_pixels:
            continue
        mean_x = sum(x for x, _ in points) / len(points)
        mean_y = sum(y for _, y in points) / len(points)
        anchor = min(points, key=lambda point: (point[0] - mean_x) ** 2 + (point[1] - mean_y) ** 2)
        item = dict(item_by_number[number])
        item["anchor"] = anchor
        item["visible_points"] = _subsample_points(points)
        item["visible_pixel_count"] = len(points)
        visible.append(item)
    return sorted(visible, key=lambda item: int(item["record"]["number"]))


def _assign_view_local_labels(visible_items, prefix):
    ordered = sorted(
        visible_items,
        key=lambda item: (
            float((item.get("anchor") or (0.0, 0.0))[1]),
            float((item.get("anchor") or (0.0, 0.0))[0]),
            int(item["record"]["number"]),
        ),
        reverse=True,
    )
    by_number = {}
    for ordinal, item in enumerate(ordered, start=1):
        labeled = dict(item)
        # S2 uses plain global numeric labels in every panel.  Stage 1 used
        # A1/B1/... local labels; users found the letters noisy for S2.
        labeled["view_label"] = f"{prefix}{ordinal}" if prefix else str(int(item["record"]["number"]))
        labeled["view_label_index"] = ordinal
        by_number[int(item["record"]["number"])] = labeled
    return [by_number[number] for number in sorted(by_number)]


def _place_inline_label_items(visible_items, min_gap=0.052):
    placed = []
    occupied = []
    ordered = sorted(visible_items, key=lambda item: float(item.get("visible_pixel_count", 0)), reverse=True)
    for item in ordered:
        anchor = item.get("anchor")
        points = list(item.get("visible_points") or [])
        if isinstance(anchor, (tuple, list)) and len(anchor) >= 2:
            points.insert(0, (float(anchor[0]), float(anchor[1])))
        if not points:
            continue
        base = points[0]
        candidates = sorted(
            [(float(x), float(y)) for x, y in points if 0.035 <= float(x) <= 0.965 and 0.045 <= float(y) <= 0.955],
            key=lambda point: (point[0] - base[0]) ** 2 + (point[1] - base[1]) ** 2,
        ) or [(min(max(base[0], 0.035), 0.965), min(max(base[1], 0.045), 0.955))]
        chosen = None
        for candidate in candidates:
            if all((candidate[0] - ox) ** 2 + (candidate[1] - oy) ** 2 >= min_gap * min_gap for ox, oy in occupied):
                chosen = candidate
                break
        if chosen is None:
            chosen = max(
                candidates,
                key=lambda point: min(((point[0] - ox) ** 2 + (point[1] - oy) ** 2) for ox, oy in occupied) if occupied else 1.0,
            )
        occupied.append(chosen)
        placed.append((item, chosen[0], chosen[1]))
    return sorted(placed, key=lambda item: str(item[0].get("view_label") or item[0]["record"]["number"]))


def _draw_inline_label_overlay(ax, visible_items, fontsize=8.6, min_gap=0.052, highlight_part_id=None):
    if not visible_items:
        return
    try:
        import matplotlib.patheffects as path_effects
    except Exception:
        path_effects = None
    overlay = _axes_overlay(ax)
    for item, x, y in _place_inline_label_items(visible_items, min_gap=min_gap):
        record = item["record"]
        label = str(item.get("view_label") or int(record["number"]))
        highlighted = highlight_part_id is not None and str(record.get("part_id")) == str(highlight_part_id)
        text = overlay.text(
            x,
            y,
            label,
            transform=overlay.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize + (1.8 if highlighted else 0.0),
            fontweight="heavy" if highlighted else "bold",
            color="#050505",
            bbox={
                "boxstyle": "round,pad=0.24,rounding_size=0.16" if highlighted else "round,pad=0.20,rounding_size=0.16",
                "facecolor": "#fff3a3" if highlighted else "white",
                "edgecolor": "#e11d48" if highlighted else part_color(record["part"], index=record["index"]),
                "linewidth": 3.2 if highlighted else 2.0,
                "alpha": 1.0 if highlighted else 0.98,
            },
            zorder=10020 if highlighted else 10005,
            clip_on=False,
        )
        if path_effects is not None:
            text.set_path_effects([path_effects.withStroke(linewidth=0.9, foreground="white")])


def _view_label_audit_entries(visible_items):
    entries = []
    for item in sorted(visible_items, key=lambda item: str(item.get("view_label", ""))):
        record = item["record"]
        entries.append({
            "view_label": str(item.get("view_label") or record["number"]),
            "part_number": int(record["number"]),
            "zero_based_index": int(record["index"]),
            "part_id": record["part_id"],
            "part_label": str(record.get("label") or f"part_{record['number']}"),
            "visible_pixel_count": int(item.get("visible_pixel_count", 0) or 0),
            "anchor_axes": [round(float(v), 4) for v in (item.get("anchor") or (0.0, 0.0))[:2]],
        })
    return entries


def _project_bbox_face_items(ax, records):
    face_items = []
    for record in records:
        vertices, _edges = _bbox_edges(record["center"], record["size"])
        for face in _bbox_faces_from_vertices(vertices):
            projected = [_project_point_axes(ax, point) for point in face]
            polygon = [(item[0], item[1]) for item in projected]
            if max(x for x, _ in polygon) < 0.0 or min(x for x, _ in polygon) > 1.0 or max(y for _, y in polygon) < 0.0 or min(y for _, y in polygon) > 1.0:
                continue
            face_items.append({
                "record": record,
                "polygon": polygon,
                "depth": -sum(item[2] for item in projected) / max(len(projected), 1),
            })
    return face_items


def _projection_rect_for_record(record, plane):
    cx, cy, cz = record["center"]
    sx, sy, sz = record["size"]
    xmin, xmax = cx - sx / 2.0, cx + sx / 2.0
    ymin, ymax = cy - sy / 2.0, cy + sy / 2.0
    zmin, zmax = cz - sz / 2.0, cz + sz / 2.0
    if plane == "front_xz":
        return xmin, xmax, zmin, zmax, -float(cy), "X", "Z"
    if plane == "side_yz":
        return ymin, ymax, zmin, zmax, float(cx), "Y", "Z"
    if plane == "top_xy":
        return xmin, xmax, ymin, ymax, float(cz), "X", "Y"
    raise ValueError(f"unknown projection plane: {plane}")


def _projection_polygon_items_for_axes(ax, rectangles):
    polygon_items = []
    for record, rx, ry, rw, rh, depth in rectangles:
        polygon = []
        for x, y in [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]:
            x_disp, y_disp = ax.transData.transform((x, y))
            x_axes, y_axes = ax.transAxes.inverted().transform((x_disp, y_disp))
            polygon.append((float(x_axes), float(y_axes)))
        polygon_items.append({"record": record, "polygon": polygon, "depth": float(depth)})
    return polygon_items


def _draw_s1_style_oblique_layout_view(ax, records, title, elev, azim, prefix, highlight_part_id=None):
    import matplotlib
    matplotlib.use("Agg")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    _set_equal_3d_for_records(ax, records)
    ax.view_init(elev=elev, azim=azim)
    fig = ax.get_figure()
    fig.canvas.draw()
    face_records = []
    for record in records:
        color = _hex_to_rgb01(part_color(record["part"], index=record["index"])) + (1.0,)
        vertices, _edges = _bbox_edges(record["center"], record["size"])
        for face in _bbox_faces_from_vertices(vertices):
            projected = [_project_point_axes(ax, point) for point in face]
            depth = -sum(item[2] for item in projected) / max(len(projected), 1)
            face_records.append((depth, record["index"], face, color, record))
    for rank, (_depth, _index, face, color, record) in enumerate(sorted(face_records, key=lambda item: (item[0], item[1]))):
        highlighted = highlight_part_id is not None and str(record.get("part_id")) == str(highlight_part_id)
        poly = Poly3DCollection(
            [face],
            facecolors=[color],
            edgecolors="#e11d48" if highlighted else (0.05, 0.05, 0.05, 0.78),
            linewidths=2.2 if highlighted else 0.55,
            alpha=1.0,
            zsort="average",
        )
        poly.set_zorder(10 + rank)
        ax.add_collection3d(poly)
    fig.canvas.draw()
    visible_items = _assign_view_local_labels(_pixel_visible_items_from_polygons(_project_bbox_face_items(ax, records), resolution=112, min_visible_pixels=6), prefix)
    _draw_inline_label_overlay(ax, visible_items, fontsize=8.6, min_gap=0.052, highlight_part_id=highlight_part_id)
    ax.set_title(title, fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel("X", labelpad=-2)
    ax.set_ylabel("Y", labelpad=-2)
    ax.set_zlabel("Z", labelpad=-2)
    ax.grid(True, alpha=0.25)
    return visible_items


def _draw_s1_style_projection_layout_view(ax, records, title, plane, prefix, highlight_part_id=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rectangles = []
    x_values = []
    y_values = []
    xlabel = ylabel = ""
    for record in records:
        px0, px1, py0, py1, depth, xlabel, ylabel = _projection_rect_for_record(record, plane)
        rx, ry, rw, rh = px0, py0, px1 - px0, py1 - py0
        rectangles.append((record, rx, ry, rw, rh, depth))
        x_values.extend([rx, rx + rw])
        y_values.extend([ry, ry + rh])
    if not rectangles:
        ax.text(0.5, 0.5, "No parts", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="bold")
        return []
    x_pad = max((max(x_values) - min(x_values)) * 0.08, 0.08)
    y_pad = max((max(y_values) - min(y_values)) * 0.08, 0.08)
    ax.set_xlim(min(x_values) - x_pad, max(x_values) + x_pad)
    ax.set_ylim(min(y_values) - y_pad, max(y_values) + y_pad)
    if plane in {"front_xz", "side_yz"}:
        ax.axhline(0.0, color="#111111", linewidth=1.2, linestyle="--", alpha=0.75)
        ax.text(0.01, 0.02, "ground z=0", transform=ax.transAxes, fontsize=9, color="#222", va="bottom")
    for record, rx, ry, rw, rh, depth in sorted(rectangles, key=lambda item: (item[5], item[3] * item[4])):
        highlighted = highlight_part_id is not None and str(record.get("part_id")) == str(highlight_part_id)
        ax.add_patch(plt.Rectangle((rx, ry), rw, rh, linewidth=2.6 if highlighted else 1.2, edgecolor="#e11d48" if highlighted else "#171717", facecolor=part_color(record["part"], index=record["index"]), alpha=1.0, zorder=10 + depth * 0.001 if highlighted else 2 + depth * 0.001))
    ax.get_figure().canvas.draw()
    visible_items = _assign_view_local_labels(_pixel_visible_items_from_polygons(_projection_polygon_items_for_axes(ax, rectangles), resolution=112, min_visible_pixels=5), prefix)
    _draw_inline_label_overlay(ax, visible_items, fontsize=8.6, min_gap=0.052, highlight_part_id=highlight_part_id)
    ax.set_title(title, fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_alpha(0.35)
    return visible_items


def render_s1_style_layout_multiview(contract, output_path, title="S1-style layout visual critic multiview", highlight_part_id=None):
    """Render layout bboxes as a six-panel readable-label VLM board.

    The PNG follows the S1 visual style (opaque boxes, inline labels,
    multiview coverage sidecar) but S2 uses plain global numeric labels instead
    of A1/B1/... per-view prefixes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = _layout_part_records(contract)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(18.8, 12.6), dpi=150, facecolor="#f7f3ea")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    view_audit = []
    solid_views = [
        ("ISO numeric labels", "", 28, -42),
        ("Left street-oblique numeric labels", "", 16, -70),
        ("Right street-oblique numeric labels", "", 16, 35),
    ]
    for panel_index, (view_title, prefix, elev, azim) in enumerate(solid_views, start=1):
        ax = fig.add_subplot(2, 3, panel_index, projection="3d", facecolor="#fbfaf5")
        visible_items = _draw_s1_style_oblique_layout_view(ax, records, view_title, elev, azim, prefix, highlight_part_id=highlight_part_id)
        entries = _view_label_audit_entries(visible_items)
        view_audit.append({"view": view_title, "view_prefix": prefix, "visible_label_numbers": [entry["part_number"] for entry in entries], "view_label_map": entries, "label_count": len(entries)})
    projection_views = [
        ("Front X-Z numeric labels", "", "front_xz"),
        ("Side Y-Z numeric labels", "", "side_yz"),
        ("Top X-Y numeric labels", "", "top_xy"),
    ]
    for panel_index, (view_title, prefix, plane) in enumerate(projection_views, start=4):
        ax = fig.add_subplot(2, 3, panel_index, facecolor="#fbfaf5")
        visible_items = _draw_s1_style_projection_layout_view(ax, records, view_title, plane, prefix, highlight_part_id=highlight_part_id)
        entries = _view_label_audit_entries(visible_items)
        view_audit.append({"view": view_title, "view_prefix": prefix, "plane": plane, "visible_label_numbers": [entry["part_number"] for entry in entries], "view_label_map": entries, "label_count": len(entries)})
    fig.text(
        0.5,
        0.012,
        "Labels are global part numbers reused consistently across all panels; JSON sidecar maps each number to part_id.",
        ha="center",
        fontsize=11,
        color="#333",
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.925, bottom=0.075, wspace=0.18, hspace=0.2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.18, facecolor=fig.get_facecolor())
    plt.close(fig)
    colors = [part_color(record["part"], index=record["index"]) for record in records]
    audit = {
        "schema": "archstudio_s2.s1_style_layout_visual_label_coverage.v1",
        "identity_contract": "global_numeric_pixel_visible_inline_labels",
        "part_count": len(records),
        "all_parts_listed_once": [record["number"] for record in records] == list(range(1, len(records) + 1)),
        "unique_color_count": len(set(colors)),
        "all_colors_unique": len(colors) == len(set(colors)),
        "legend_entries": [
            {
                "part_number": int(record["number"]),
                "zero_based_index": int(record["index"]),
                "part_id": record["part_id"],
                "color_hex": part_color(record["part"], index=record["index"]),
                "label": str(record.get("label") or f"part_{record['number']}"),
            }
            for record in records
        ],
        "views": view_audit,
        "total_view_label_count": sum(int(item["label_count"]) for item in view_audit),
        "highlight_part_id": str(highlight_part_id or ""),
        "note": "S2 layout evidence keeps the S1 six-panel opaque-box visual style but uses plain global numeric labels instead of A/B/C per-view prefixes. Optional highlight uses the same S1 layout rendering, not an extra dark bbox proxy.",
    }
    audit_path = output_path.with_name(output_path.stem + "_label_coverage.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(output_path)


def compose_image_grid(image_items, output_path, title, columns=3, tile_size=(620, 420), background="#f4efe5"):
    """Tile existing images into one labeled PNG suitable for a single VLM image input."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    valid = []
    for label, path in image_items:
        candidate = Path(path)
        if candidate.exists() and candidate.is_file():
            valid.append((str(label), candidate))
    if not valid:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_w, tile_h = tile_size
    label_h = 42
    margin = 24
    gap = 18
    columns = max(1, int(columns))
    rows = int(math.ceil(len(valid) / float(columns)))
    width = columns * tile_w + (columns - 1) * gap + 2 * margin
    height = 68 + rows * (tile_h + label_h) + (rows - 1) * gap + margin
    board = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(board)
    draw.rectangle((0, 0, width, 62), fill="#eaf2fb")
    draw.text((margin, 18), str(title), fill="#172033", font=font)
    for index, (label, source_path) in enumerate(valid, start=1):
        row = (index - 1) // columns
        col = (index - 1) % columns
        x = margin + col * (tile_w + gap)
        y = 72 + row * (tile_h + label_h + gap)
        draw.rounded_rectangle((x, y, x + tile_w, y + tile_h + label_h), radius=12, fill="#fffdf7", outline="#c9bda5")
        draw.rectangle((x + 10, y + 10, x + tile_w - 10, y + tile_h - 10), fill="#f7fafc")
        try:
            image = Image.open(source_path).convert("RGB")
            image.thumbnail((tile_w - 20, tile_h - 20), Image.LANCZOS)
            px = x + (tile_w - image.width) // 2
            py = y + 10 + (tile_h - 20 - image.height) // 2
            board.paste(image, (px, py))
        except Exception as exc:
            draw.text((x + 18, y + 22), f"failed to load image: {exc!r}"[:120], fill="#ffb4ab", font=font)
        draw.rounded_rectangle((x + 12, y + tile_h + 8, x + tile_w - 12, y + tile_h + label_h - 8), radius=7, fill="#2f4b67")
        draw.text((x + 22, y + tile_h + 18), f"{index:02d}. {label}"[:110], fill="#ffffff", font=font)
    board.save(output_path)
    return str(output_path)

def render_layout_xy(contract, output_path, highlight_part_id=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = contract["parts"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    for index, part in enumerate(parts):
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        x = center[0] - size[0] / 2
        y = center[1] - size[1] / 2
        is_highlight = part["part_id"] == highlight_part_id
        target_class = part.get("target_class", "object")
        color = (
            part_color(part, index=index)
            if target_class == "open_semantic_part"
            else CLASS_COLORS.get(target_class, CLASS_COLORS["object"])
        )
        rect = Rectangle(
            (x, y),
            size[0],
            size[1],
            linewidth=1.8 if is_highlight else 0.5,
            edgecolor="#111" if is_highlight else color,
            facecolor=color if is_highlight else "#dddddd",
            alpha=0.75 if is_highlight else 0.22,
        )
        ax.add_patch(rect)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Layout XY" + (" highlight" if highlight_part_id else ""))
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def render_layout_projection(contract, output_path, axes=("x", "z"), highlight_part_id=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    axis_index = {"x": 0, "y": 1, "z": 2}
    i, j = axis_index[axes[0]], axis_index[axes[1]]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    for index, part in enumerate(contract["parts"]):
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        x = center[i] - size[i] / 2
        y = center[j] - size[j] / 2
        is_highlight = part["part_id"] == highlight_part_id
        target_class = part.get("target_class", "object")
        color = (
            part_color(part, index=index)
            if target_class == "open_semantic_part"
            else CLASS_COLORS.get(target_class, CLASS_COLORS["object"])
        )
        ax.add_patch(
            Rectangle(
                (x, y),
                size[i],
                size[j],
                linewidth=1.8 if is_highlight else 0.45,
                edgecolor="#111" if is_highlight else color,
                facecolor=color,
                alpha=0.75 if is_highlight else 0.24,
            )
        )
    ax.set_aspect("equal", "box")
    ax.set_xlabel(axes[0].upper())
    ax.set_ylabel(axes[1].upper())
    ax.set_title(f"Layout {axes[0].upper()}{axes[1].upper()} projection")
    ax.autoscale_view()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def _shorten_part_prompt(prompt):
    text = " ".join(str(prompt or "").split())
    replacements = [
        ("ONE single closed rectangular window 3D asset render, exactly one object, simple window frame with closed glass panes, visible thin depth, not open, not broken, no wall, no sill", "closed rectangular window; thin frame/glass; no wall/sill"),
        ("ONE single closed narrow door 3D asset render, exactly one object, one closed door slab in one frame with one handle, visible thin depth, not open, not double doors, no wall", "closed narrow door; slab+frame+handle; no wall"),
        ("ONE detached triangular prism gable roof 3D asset render, exactly one object, simple sloped roof cap with eaves, no walls, no columns, no base", "detached gable roof; sloped cap/eaves; no wall/base"),
        ("ONE plain solid rectangular wall block 3D asset render, exactly one object, flat plaster slab with subtle surface, no border frame, no windows, no doors", "plain wall block; no windows/doors"),
        ("isolated single component", "isolated"),
    ]
    for source, target in replacements:
        text = text.replace(source, target)
    stop_phrases = [
        ", centered front view",
        ", no background scene",
        ", only one object",
    ]
    for phrase in stop_phrases:
        if phrase in text:
            text = text.split(phrase)[0]
    return text[:90].rstrip(" ,")


def render_layout_text_callouts(
    contract,
    output_path,
    axes=("x", "z"),
    title="Layout text callouts",
):
    """Render a layout projection with per-part leader lines and prompt labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    axis_index = {"x": 0, "y": 1, "z": 2}
    i, j = axis_index[axes[0]], axis_index[axes[1]]
    parts = list(contract["parts"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(17, 11), dpi=150, facecolor="#f7f7f5")
    ax.set_facecolor("#fbfbf8")
    projected = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for index, part in enumerate(parts):
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        x = center[i] - size[i] / 2
        y = center[j] - size[j] / 2
        color = part_color(part, index=index)
        ax.add_patch(
            Rectangle(
                (x, y),
                size[i],
                size[j],
                linewidth=1.0,
                edgecolor=color,
                facecolor=color,
                alpha=0.22,
            )
        )
        cx, cy = center[i], center[j]
        ax.scatter([cx], [cy], s=28, color=color, zorder=5)
        ax.text(
            cx,
            cy,
            str(index + 1),
            fontsize=7,
            fontweight="bold",
            color="#111",
            ha="center",
            va="center",
            zorder=6,
        )
        projected.append((index, part, cx, cy, color))
        min_x = min(min_x, x)
        max_x = max(max_x, x + size[i])
        min_y = min(min_y, y)
        max_y = max(max_y, y + size[j])

    if not projected:
        return None

    x_span = max(max_x - min_x, 1e-6)
    y_span = max(max_y - min_y, 1e-6)
    left_x = min_x - x_span * 0.78
    right_x = max_x + x_span * 0.78
    sorted_parts = sorted(projected, key=lambda item: (-item[3], item[2]))
    left_items = []
    right_items = []
    for item in sorted_parts:
        if item[2] < (min_x + max_x) / 2:
            left_items.append(item)
        else:
            right_items.append(item)

    def label_positions(items):
        if not items:
            return []
        top = max_y + y_span * 0.14
        bottom = min_y - y_span * 0.14
        if len(items) == 1:
            return [(items[0], (top + bottom) / 2)]
        step = (top - bottom) / (len(items) - 1)
        return [(item, top - idx * step) for idx, item in enumerate(items)]

    for side, items in (("left", left_items), ("right", right_items)):
        label_x = left_x if side == "left" else right_x
        ha = "right" if side == "left" else "left"
        for item, label_y in label_positions(items):
            index, part, cx, cy, color = item
            prompt = _shorten_part_prompt(part.get("part_description") or part.get("open_vocab_label") or part.get("part_prompt") or part.get("target_prompt"))
            label_class = part.get("open_vocab_label") or part.get("target_class", "object")
            label = "{}. {}: {}".format(index + 1, label_class, prompt)
            wrapped = "\n".join(textwrap.wrap(label, width=34))
            elbow_x = min_x - x_span * 0.10 if side == "left" else max_x + x_span * 0.10
            ax.plot([cx, elbow_x, label_x], [cy, label_y, label_y], color=color, linewidth=0.9, alpha=0.9)
            ax.text(
                label_x,
                label_y,
                wrapped,
                ha=ha,
                va="center",
                fontsize=6.6,
                color="#161616",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.8,
                    "alpha": 0.94,
                },
            )

    ax.set_aspect("equal", "box")
    ax.set_xlabel(axes[0].upper())
    ax.set_ylabel(axes[1].upper())
    ax.set_title(title, fontsize=13)
    ax.set_xlim(left_x - x_span * 0.04, right_x + x_span * 0.04)
    ax.set_ylim(min_y - y_span * 0.22, max_y + y_span * 0.22)
    ax.grid(True, color="#e2e2df", linewidth=0.5, alpha=0.65)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(output_path)


def _bbox_edges(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    corners = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return corners, edges


def render_layout_3d_boxes(
    contract,
    output_path,
    elev=25,
    azim=42,
    title="3D layout bbox overview",
    figsize=(6, 5),
    transform=None,
    highlight_part_id=None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=150, facecolor="#101318")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#101318")
    xs, ys, zs = [], [], []
    for index, part in enumerate(contract["parts"]):
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        color = part_color(part, index=index)
        highlighted = highlight_part_id is not None and str(part.get("part_id")) == str(highlight_part_id)
        line_width = 2.4 if highlighted else (0.55 if highlight_part_id is not None else 0.9)
        alpha = 1.0 if highlighted else (0.18 if highlight_part_id is not None else 0.95)
        corners, edges = _bbox_edges(center, size)
        if transform is not None:
            corners = [transform(corner) for corner in corners]
        for a, b in edges:
            x = [corners[a][0], corners[b][0]]
            y = [corners[a][1], corners[b][1]]
            z = [corners[a][2], corners[b][2]]
            ax.plot(x, y, z, color=color, linewidth=line_width, alpha=alpha)
        xs.extend(c[0] for c in corners)
        ys.extend(c[1] for c in corners)
        zs.extend(c[2] for c in corners)
    _set_equal_3d(ax, xs, ys, zs)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title, color="#f2f2f2")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(output_path)



def render_multi_obj_meshes(
    obj_items,
    output_path,
    title=None,
    elev=24,
    azim=42,
    figsize=(8, 5.5),
    max_faces_per_mesh=12000,
    transform=None,
):
    """Render multiple OBJ meshes together with per-mesh colors.

    Use the same projected polygon renderer as single-part views.  This is a
    true face render (not vertex/point sampling) while staying much faster than
    Matplotlib's 3D Poly3DCollection for many Hunyuan parts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    prepared = []
    global_min = [float("inf"), float("inf"), float("inf")]
    global_max = [float("-inf"), float("-inf"), float("-inf")]

    for item in obj_items:
        obj_path = item.get("obj_path")
        color = item.get("color", CLASS_COLORS["object"])
        alpha = float(item.get("alpha", 0.985))
        if not obj_path:
            continue
        mesh_sample = read_obj_mesh_sample(obj_path, max_faces=max_faces_per_mesh)
        mesh_sample = _transform_mesh_sample(mesh_sample, transform)
        polygons = mesh_sample["polygons"]
        if not polygons:
            continue
        prepared.append((polygons, color, alpha, mesh_sample["bounds_min"], mesh_sample["bounds_max"]))
        for axis in range(3):
            global_min[axis] = min(global_min[axis], mesh_sample["bounds_min"][axis])
            global_max[axis] = max(global_max[axis], mesh_sample["bounds_max"][axis])

    if not prepared:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=150, facecolor="#f7fafc")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#f7fafc")

    global_projected = [
        _project_point(corner, *_view_basis(elev, azim))
        for corner in _bounds_corners(tuple(global_min), tuple(global_max))
    ]
    xs = [point[0] for point in global_projected]
    ys = [point[1] for point in global_projected]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    for polygons, color, alpha, bounds_min, bounds_max in prepared:
        projected, face_colors, _ = _project_mesh_polygons(
            polygons,
            bounds_min,
            bounds_max,
            elev=elev,
            azim=azim,
            color=color,
        )
        if alpha < 0.985:
            face_colors = [(r, g, b, min(a, alpha)) for r, g, b, a in face_colors]
        collection = PolyCollection(
            projected,
            facecolors=face_colors,
            edgecolors=(0.18, 0.22, 0.28, 0.0),
            linewidths=0.0,
            closed=True,
        )
        ax.add_collection(collection)

    pad = max(max_x - min_x, max_y - min_y) * 0.04 or 0.1
    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, color="#172033")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(output_path)


def render_contract_part_assembly(
    contract,
    run_dir,
    output_path,
    title=None,
    elev=24,
    azim=42,
    figsize=(8, 5.5),
    max_faces_per_mesh=12000,
    include_placeholders=True,
    placeholder_alpha=0.24,
):
    obj_items = []
    run_dir = Path(run_dir)
    for index, part in enumerate(contract["parts"]):
        part_id = part["part_id"]
        obj_path = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        used_placeholder = False
        if not obj_path.exists():
            if include_placeholders:
                obj_path = run_dir / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj"
                used_placeholder = obj_path.exists()
            else:
                continue
        if not obj_path.exists():
            obj_path = run_dir / "parts" / part_id / f"{part_id}.obj"
        if not obj_path.exists():
            continue
        obj_items.append({
            "obj_path": str(obj_path),
            "color": part_color(part, index=index),
            "alpha": placeholder_alpha if used_placeholder else 0.985,
        })
    return render_multi_obj_meshes(
        obj_items,
        output_path,
        title=title,
        elev=elev,
        azim=azim,
        figsize=figsize,
        max_faces_per_mesh=max_faces_per_mesh,
        transform=None,
    )


def render_contract_part_assembly_views(
    contract,
    run_dir,
    output_dir,
    prefix,
    title="Assembly",
    max_faces_per_mesh=12000,
    include_placeholders=True,
    placeholder_alpha=0.24,
    views=None,
):
    outputs = {}
    view_map = views or DEFAULT_3D_VIEWS
    for view_name, (elev, azim) in view_map.items():
        rendered = render_contract_part_assembly(
            contract,
            run_dir,
            Path(output_dir) / f"{prefix}_{view_name}.png",
            title=f"{title} - {view_name}",
            elev=elev,
            azim=azim,
            figsize=(8, 5.5),
            max_faces_per_mesh=max_faces_per_mesh,
            include_placeholders=include_placeholders,
            placeholder_alpha=placeholder_alpha,
        )
        if rendered:
            outputs[view_name] = rendered
    return outputs


def export_contract_part_assembly_glb(
    contract,
    run_dir,
    output_path,
    target_faces_per_part=None,
    max_vertices_per_part=None,
    include_placeholders=True,
    placeholder_alpha=96,
):
    """Export a lightweight colored assembly GLB for interactive web viewing."""
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primitives = []
    for index, part in enumerate(contract["parts"]):
        part_id = part["part_id"]
        obj_path = run_dir / "assemblies" / "parts" / f"{part_id}.obj"
        used_placeholder = False
        if not obj_path.exists():
            if include_placeholders:
                obj_path = run_dir / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj"
                used_placeholder = obj_path.exists()
            else:
                continue
        if not obj_path.exists():
            obj_path = run_dir / "parts" / part_id / f"{part_id}.obj"
        if not obj_path.exists():
            continue
        vertices, triangles = _read_obj_full_mesh(obj_path)
        if not vertices or not triangles:
            continue
        if max_vertices_per_part:
            vertices, triangles = _weld_vertices_by_grid(
                vertices,
                triangles,
                max_vertices=max_vertices_per_part,
            )
        elif target_faces_per_part:
            # Backward-compatible fallback for older callers. Avoid this for
            # report assembly viewers because sparse stride sampling makes a
            # real mesh look like a point/triangle cloud.
            pass
        vertices, triangles = _compact_indexed_mesh(vertices, triangles)
        if not vertices or not triangles:
            continue
        color = part_color(part, index=index)
        primitives.append({
            "name": part_id,
            "vertices": _transform_vertices_for_glb(vertices),
            "triangles": triangles,
            "rgba": _hex_to_rgba255(
                color,
                alpha=int(placeholder_alpha) if used_placeholder else 255,
            ),
        })

    return _write_glb(primitives, output_path)


def render_layout_3d_box_views(
    contract,
    output_dir,
    prefix,
    title="3D layout",
    transform=None,
    highlight_part_id=None,
    views=None,
):
    outputs = {}
    view_map = views or DEFAULT_3D_VIEWS
    for view_name, (elev, azim) in view_map.items():
        rendered = render_layout_3d_boxes(
            contract,
            Path(output_dir) / f"{prefix}_{view_name}.png",
            elev=elev,
            azim=azim,
            title=f"{title} - {view_name}",
            figsize=(6, 4.8),
            transform=transform,
            highlight_part_id=highlight_part_id,
        )
        if rendered:
            outputs[view_name] = rendered
    return outputs


def export_layout_3d_boxes_glb(contract, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    primitives = []
    for index, part in enumerate(contract["parts"]):
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        vertices, triangles = _box_vertices_and_triangles(center, size)
        color = part_color(part, index=index)
        primitives.append({
            "name": part["part_id"],
            "vertices": _transform_vertices_for_glb(vertices),
            "triangles": triangles,
            "rgba": _hex_to_rgba255(color, alpha=110),
        })
    return _write_glb(primitives, output_path)
