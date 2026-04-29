"""Part-level prompt generation for BuildingBlock mesh conditioning."""

import hashlib


NEGATIVE_PROMPT = (
    "full building, building facade, exterior elevation, street, city, room, "
    "people, person, text, watermark, logo, label, blurry, low quality, "
    "multiple objects, repeated objects, duplicate object, pair, two objects, "
    "four-panel sheet, contact sheet, split screen, double door, double doors, "
    "three objects, triptych, collage, grid, tiles, background clutter, "
    "landscape, sky, grass, trees, ground, floor, background, scene background, "
    "pedestal, platform, stand, base, support, supporting column, shadow catcher, "
    "neighboring parts, attached wall, house body, border frame, extra windows, extra doors, "
    "columns, posts, pavilion, gazebo, heavy black outline, line art, blueprint, technical drawing"
)

CLASS_PROMPT_TEMPLATES = {
    "window": (
        "ONE single window 3D asset render, exactly one object in the image, "
        "a simple rectangular window frame with glass panes"
    ),
    "door": (
        "ONE single narrow closed door 3D asset render, exactly one object, "
        "one door slab in one frame with one handle, not double doors"
    ),
    "roof": (
        "ONE detached triangular prism gable roof 3D asset render, exactly one object, "
        "simple sloped roof shell cap only, roof surface with eaves, no walls, no columns"
    ),
    "wall": (
        "ONE plain solid rectangular wall block 3D asset render, exactly one object, "
        "flat plaster slab with subtle surface, no border frame, no windows, no doors"
    ),
    "balcony": (
        "ONE single balcony 3D asset render, exactly one object in the image, "
        "small platform slab with front railing"
    ),
    "chimney": (
        "ONE single chimney 3D asset render, exactly one object in the image, "
        "vertical rectangular chimney stack with top cap"
    ),
}


def normalize_prompt_text(value):
    return " ".join(str(value).strip().split())


def _shape_phrase(size):
    if len(size) != 3:
        return ""
    sx, sy, sz = [float(value) for value in size]
    footprint = max(sx, sy, 1e-6)
    thickness = min(sx, sy)
    if sz > footprint * 1.5:
        return "tall proportions"
    if thickness < footprint * 0.18:
        return "thin shallow depth"
    if footprint > sz * 1.8:
        return "wide proportions"
    return "compact proportions"


def build_part_prompt(part):
    target_class = part.get("target_class", "object")
    base = CLASS_PROMPT_TEMPLATES.get(
        target_class,
        "one complete freestanding architectural {} asset, single object only".format(
            str(target_class).replace("_", " ")
        ),
    )
    size = part.get("actor_size", part.get("bbox", {}).get("size", []))
    shape_phrase = _shape_phrase(size)
    shape_text = ", {}".format(shape_phrase) if shape_phrase else ""
    return normalize_prompt_text(
        "{}, transparent background look, isolated object cutout, centered, front view,"
        " no ground plane, no pedestal, no support, studio product render, only one object{}".format(
            base,
            shape_text,
        )
    )


def prompt_hash(prompt, negative_prompt=NEGATIVE_PROMPT):
    payload = "{}\n---\n{}".format(
        normalize_prompt_text(prompt),
        normalize_prompt_text(negative_prompt),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
