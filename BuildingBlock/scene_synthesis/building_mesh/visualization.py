"""Lightweight visualization helpers for BuildingBlock-Hunyuan reports."""

import math
import json
import struct
from pathlib import Path
import random


CLASS_COLORS = {
    "wall": "#4C78A8",
    "window": "#54A24B",
    "door": "#F58518",
    "roof": "#E45756",
    "balcony": "#B279A2",
    "chimney": "#9D755D",
    "object": "#BAB0AC",
}


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


def read_obj_mesh_sample(path, max_faces=12000):
    vertices = []
    sampled_faces = []
    rng = random.Random(5678)
    faces_seen = 0
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
            elif line.startswith("f "):
                face = _parse_obj_face_indices(line.split()[1:], len(vertices))
                if face is None:
                    continue
                faces_seen += 1
                if len(sampled_faces) < max_faces:
                    sampled_faces.append(face)
                else:
                    replacement_index = rng.randrange(faces_seen)
                    if replacement_index < max_faces:
                        sampled_faces[replacement_index] = face

    polygons = []
    for face in sampled_faces:
        try:
            polygons.append([vertices[index] for index in face])
        except IndexError:
            continue

    if not vertices:
        bounds_min = [0.0, 0.0, 0.0]
        bounds_max = [0.0, 0.0, 0.0]

    return {
        "polygons": polygons,
        "bounds_min": tuple(bounds_min),
        "bounds_max": tuple(bounds_max),
    }


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
        ax.set_title(title, fontsize=10, color="#f2f2f2")
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
    )
    if not projected:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=150, facecolor="#101318")
    ax.set_facecolor("#101318")
    collection = PolyCollection(
        projected,
        facecolors=face_colors,
        edgecolors=(0.04, 0.05, 0.07, 0.18),
        linewidths=0.05,
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
        ax.set_title(title, fontsize=10, color="#f2f2f2")
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


def _project_mesh_polygons(polygons, bounds_min, bounds_max, elev, azim, color):
    right, up, view_dir = _view_basis(elev, azim)
    base_rgb = _hex_to_rgb01(color)
    light_dir = _normalize((0.35, -0.45, 0.82))
    projected_with_depth = []

    for polygon in polygons:
        if len(polygon) < 3:
            continue
        projected = [_project_point(point, right, up, view_dir) for point in polygon]
        points_2d = [(point[0], point[1]) for point in projected]
        depth = sum(point[2] for point in projected) / len(projected)
        normal = _normalize(_cross(_sub(polygon[1], polygon[0]), _sub(polygon[2], polygon[0])))
        shade = 0.34 + 0.66 * abs(_dot(normal, light_dir))
        face_color = (
            min(base_rgb[0] * shade, 1.0),
            min(base_rgb[1] * shade, 1.0),
            min(base_rgb[2] * shade, 1.0),
            0.96,
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


def render_obj_mesh_views(obj_path, output_dir, prefix, title="Mesh", max_faces=12000):
    mesh_sample = read_obj_mesh_sample(obj_path, max_faces=max_faces)
    outputs = {}
    for view_name, (elev, azim) in DEFAULT_3D_VIEWS.items():
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
    if outputs:
        return outputs
    return render_obj_point_views(obj_path, output_dir, prefix, title=title)


def render_layout_xy(contract, output_path, highlight_part_id=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = contract["parts"]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    for part in parts:
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        x = center[0] - size[0] / 2
        y = center[1] - size[1] / 2
        is_highlight = part["part_id"] == highlight_part_id
        color = CLASS_COLORS.get(part.get("target_class", "object"), CLASS_COLORS["object"])
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
    for part in contract["parts"]:
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        x = center[i] - size[i] / 2
        y = center[j] - size[j] / 2
        is_highlight = part["part_id"] == highlight_part_id
        color = CLASS_COLORS.get(part.get("target_class", "object"), CLASS_COLORS["object"])
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
    for part in contract["parts"]:
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        color = CLASS_COLORS.get(part.get("target_class", "object"), CLASS_COLORS["object"])
        corners, edges = _bbox_edges(center, size)
        for a, b in edges:
            x = [corners[a][0], corners[b][0]]
            y = [corners[a][1], corners[b][1]]
            z = [corners[a][2], corners[b][2]]
            ax.plot(x, y, z, color=color, linewidth=0.9, alpha=0.95)
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



def render_multi_obj_meshes(obj_items, output_path, title=None, elev=24, azim=42, figsize=(8, 5.5), max_faces_per_mesh=12000):
    """Render multiple OBJ meshes together with per-mesh colors.

    Use a true 3D Poly3DCollection renderer so the result reads like a solid
    mesh render rather than a projected point-cloud/wireframe impression.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    prepared = []
    global_min = [float("inf"), float("inf"), float("inf")]
    global_max = [float("-inf"), float("-inf"), float("-inf")]
    light_dir = _normalize((0.38, -0.42, 0.82))

    for item in obj_items:
        obj_path = item.get("obj_path")
        color = item.get("color", CLASS_COLORS["object"])
        if not obj_path:
            continue
        mesh_sample = read_obj_mesh_sample(obj_path, max_faces=max_faces_per_mesh)
        polygons = mesh_sample["polygons"]
        if not polygons:
            continue
        base_rgb = _hex_to_rgb01(color)
        face_colors = []
        for polygon in polygons:
            if len(polygon) < 3:
                face_colors.append((*base_rgb, 0.98))
                continue
            normal = _normalize(_cross(_sub(polygon[1], polygon[0]), _sub(polygon[2], polygon[0])))
            shade = 0.30 + 0.70 * abs(_dot(normal, light_dir))
            face_colors.append((
                min(base_rgb[0] * shade, 1.0),
                min(base_rgb[1] * shade, 1.0),
                min(base_rgb[2] * shade, 1.0),
                0.985,
            ))
        prepared.append((polygons, face_colors, mesh_sample["bounds_min"], mesh_sample["bounds_max"]))
        for axis in range(3):
            global_min[axis] = min(global_min[axis], mesh_sample["bounds_min"][axis])
            global_max[axis] = max(global_max[axis], mesh_sample["bounds_max"][axis])

    if not prepared:
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=figsize, dpi=150, facecolor="#101318")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#101318")

    for polygons, face_colors, _, _ in prepared:
        collection = Poly3DCollection(
            polygons,
            facecolors=face_colors,
            edgecolors=(0.02, 0.03, 0.05, 0.10),
            linewidths=0.03,
        )
        ax.add_collection3d(collection)

    _set_equal_3d_bounds(ax, tuple(global_min), tuple(global_max))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, color="#f2f2f2")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return str(output_path)


def render_contract_part_assembly(contract, run_dir, output_path, title=None, elev=24, azim=42, figsize=(8, 5.5), max_faces_per_mesh=12000):
    obj_items = []
    run_dir = Path(run_dir)
    for part in contract["parts"]:
        part_id = part["part_id"]
        obj_path = run_dir / "parts" / part_id / f"{part_id}.obj"
        if not obj_path.exists():
            continue
        obj_items.append({
            "obj_path": str(obj_path),
            "color": CLASS_COLORS.get(part.get("target_class", "object"), CLASS_COLORS["object"]),
        })
    return render_multi_obj_meshes(
        obj_items,
        output_path,
        title=title,
        elev=elev,
        azim=azim,
        figsize=figsize,
        max_faces_per_mesh=max_faces_per_mesh,
    )


def render_contract_part_assembly_views(contract, run_dir, output_dir, prefix, title="Assembly", max_faces_per_mesh=12000):
    outputs = {}
    for view_name, (elev, azim) in DEFAULT_3D_VIEWS.items():
        rendered = render_contract_part_assembly(
            contract,
            run_dir,
            Path(output_dir) / f"{prefix}_{view_name}.png",
            title=f"{title} - {view_name}",
            elev=elev,
            azim=azim,
            figsize=(8, 5.5),
            max_faces_per_mesh=max_faces_per_mesh,
        )
        if rendered:
            outputs[view_name] = rendered
    return outputs


def export_contract_part_assembly_glb(contract, run_dir, output_path, target_faces_per_part=None):
    """Export a lightweight colored assembly GLB for interactive web viewing."""
    run_dir = Path(run_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primitives = []
    for part in contract["parts"]:
        part_id = part["part_id"]
        obj_path = run_dir / "parts" / part_id / f"{part_id}.obj"
        if not obj_path.exists():
            continue
        vertices, triangles = _read_obj_full_mesh(obj_path)
        if not vertices or not triangles:
            continue
        if target_faces_per_part and len(triangles) > target_faces_per_part:
            stride = max(1, len(triangles) // target_faces_per_part)
            triangles = triangles[::stride][:target_faces_per_part]
        vertices, triangles = _compact_indexed_mesh(vertices, triangles)
        if not vertices or not triangles:
            continue
        color = CLASS_COLORS.get(part.get("target_class", "object"), CLASS_COLORS["object"])
        primitives.append({
            "name": part_id,
            "vertices": vertices,
            "triangles": triangles,
            "rgba": _hex_to_rgba255(color, alpha=255),
        })

    return _write_glb(primitives, output_path)


def render_layout_3d_box_views(contract, output_dir, prefix, title="3D layout"):
    outputs = {}
    for view_name, (elev, azim) in DEFAULT_3D_VIEWS.items():
        rendered = render_layout_3d_boxes(
            contract,
            Path(output_dir) / f"{prefix}_{view_name}.png",
            elev=elev,
            azim=azim,
            title=f"{title} - {view_name}",
            figsize=(6, 4.8),
        )
        if rendered:
            outputs[view_name] = rendered
    return outputs


def export_layout_3d_boxes_glb(contract, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    primitives = []
    for part in contract["parts"]:
        center = part["bbox"]["center"]
        size = part["bbox"]["size"]
        vertices, triangles = _box_vertices_and_triangles(center, size)
        color = CLASS_COLORS.get(part.get("target_class", "object"), CLASS_COLORS["object"])
        primitives.append({
            "name": part["part_id"],
            "vertices": vertices,
            "triangles": triangles,
            "rgba": _hex_to_rgba255(color, alpha=110),
        })
    return _write_glb(primitives, output_path)
