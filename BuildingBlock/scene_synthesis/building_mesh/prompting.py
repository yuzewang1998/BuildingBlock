"""Part-level prompt generation for BuildingBlock mesh conditioning."""

import hashlib
import re


PROMPT_POLICY_VERSION = "archstudio_s2_part_focus_v10_single_component"


RELATIONAL_SPLIT_MARKERS = (
    " enclosing ",
    " wrapping ",
    " spanning ",
    " running ",
    " projecting ",
    " projects ",
    " covering ",
    " covers ",
    " aligned ",
    " forming ",
    " forms ",
    " creates ",
    " behind ",
    " below ",
    " above ",
    " between ",
    " around ",
    " along ",
    " across ",
    " through ",
    " for ",
    " of a ",
    " of an ",
    " of the ",
    " in a ",
    " in an ",
    " in the ",
    " at a ",
    " at an ",
    " at the ",
    " on a ",
    " on an ",
    " on the ",
)


CONTEXT_WORDS = (
    "courtyard",
    "building",
    "museum",
    "market hall",
    "temple",
    "apartment",
    "library",
    "chapel",
    "airport",
    "factory",
    "scene",
    "overall",
    "surrounding",
    "adjacent",
)

DIRECTIONAL_PREFIX_WORDS = {
    "north",
    "south",
    "east",
    "west",
    "left",
    "right",
    "front",
    "rear",
    "back",
}


NEGATIVE_PROMPT = (
    "full building, entire building, whole building, complete building, architectural complex, building complex, "
    "complete courtyard, U-shaped courtyard, surrounding courtyard, multiple wings, surrounding galleries, "
    "complete museum, complete market hall, complete temple, complete chapel, complete factory, "
    "building facade, exterior elevation, facade composition, street, city, room, "
    "people, person, text, words, letters, signage, signboard, shop sign, watermark, logo, label, blurry, low quality, "
    "multiple objects, repeated objects, duplicate object, pair, two objects, "
    "four-panel sheet, contact sheet, split screen, three objects, triptych, collage, grid, tiles, background clutter, "
    "landscape, sky, grass, trees, ground, floor, background, scene background, room, interior, exterior wall, "
    "background plane, back plane, backing board, flat backing board, wall panel, rectangular panel behind the object, "
    "white board, black board, display board, mounting plate, mat board, poster board, slab behind object, "
    "pedestal, platform, stand, base, plinth, podium, support, supporting column, footing, foundation, sill, ledge, "
    "shadow catcher, cast shadow, table, shelf, tile floor, carpet, wallpaper, texture backdrop, patterned backdrop, "
    "neighboring parts, attached wall, house body, border frame, contextual architecture, "
    "heavy black outline, line art, blueprint, technical drawing"
)

BACKGROUND_CONTROL_TEXT = (
    "on a seamless pure white #FFFFFF background that is not part of the object geometry, "
    "the background must remain empty negative space with no physical panel, no backing board, no wall plane, "
    "no rectangular sheet behind or around the object, no mounting plate, no floor, no shadow catcher; "
    "the object itself should fill 88 to 95 percent of the image frame, nearly touching the image bounds, "
    "crop tightly around the single object while keeping the entire object visible; "
    "only the architectural component should be reconstructed as 3D, never the background"
)

GLOBAL_CIVIC_STYLE_TEXT = (
    "realistic architectural asset material and form consistent with the component description, "
    "matte physically plausible surface with subtle weathering and restrained architectural detail, "
    "single component, no cartoon style, no toy plastic, no random colorful materials"
)

ORTHOGRAPHIC_ASSET_TEXT = (
    "orthographic product view, neutral studio lighting, crisp silhouette, centered, front-facing asset preview, "
    "the component fills most of the frame without being cut off"
)

GENERIC_OBJECT_ONLY_TEXT = (
    "single isolated object cutout only, no base, no pedestal, no plinth, no podium, no platform, "
    "no display stand, no support stand, no support column, no footing, no foundation, no sill, no ledge, "
    "no long bottom rail added just to fill the image, no wide flat board or strip attached under the object; "
    "do not add props that make the object stand up, do not add any extra stabilizing geometry, "
    "the silhouette must be the component itself rather than component plus base"
)

WALL_FACADE_PROMPT_TEMPLATE = (
    "single non-repeating exterior facade elevation texture map for one architectural wall face, "
    "orthographic front elevation image mapped onto a flat rectangular wall plane, fills the entire image, "
    "realistic light warm limestone and pale civic stone facade, varied masonry blocks, carved stone joints, "
    "subtle weathering, stains, chipped edges, uneven stone color, low relief architectural surface detail, "
    "photorealistic but still a flat UV texture image, no repeated tile pattern, no seamless material swatch"
)

WALL_FACADE_NEGATIVE_PROMPT = (
    "3D object render, cube, box, floating block, product render, perspective view, street scene, sky, ground, floor, "
    "people, cars, trees, surrounding buildings, full building in environment, repeated tile texture, seamless pattern, "
    "tiny regular bricks repeated mechanically, wallpaper, contact sheet, border, frame, label, text, logo, shadow catcher"
)

WALL_TEXTURE_POLICY_TEXT = (
    "wall placeholders are fixed layout bbox geometry, so this image must be a single detailed exterior face texture, "
    "not a wall-shaped 3D component and not a small material tile; it will be projected once onto the detected outside face"
)

REFERENCE_FRAME_TEXT = (
    "orientation constraint: the top of the image is the architectural top, the bottom of the image is the "
    "architectural bottom near the ground, and the visible face is the outward-facing exterior side of the component; "
    "do not rotate the component sideways or upside down"
)

FACADE_TEXTURE_FRAME_TEXT = (
    "orientation constraint: the top edge of the image is architectural up, the bottom edge is nearest the ground, "
    "horizontal facade courses and stone joints run left-to-right across the image, and the image shows the outward "
    "exterior face; never rotate this texture map 90 degrees or upside down"
)


def normalize_prompt_text(value):
    return " ".join(str(value).strip().split())


def _strip_prompt_prefix(text):
    text = normalize_prompt_text(text)
    prefixes = (
        "open-vocabulary architectural component:",
        "part-only open-vocabulary architectural component:",
        "create one architectural component:",
        "one architectural component:",
        "3d mesh of",
    )
    lowered = text.lower()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return normalize_prompt_text(text[len(prefix):])
    return text


def component_source_description(part):
    for key in (
        "part_description",
        "open_vocab_label",
        "source_part_name",
        "source_actor_label",
        "target_prompt",
        "target_class",
    ):
        value = part.get(key)
        if value not in (None, "", []):
            return _strip_prompt_prefix(str(value).replace("_", " "))
    return "architectural component"


def component_core_description(part):
    """Return the part noun phrase without layout placement/context clauses.

    Stage-1 open-semantic labels often mix the component with placement text,
    e.g. ``north gallery bar enclosing the back of a U-shaped museum courtyard``.
    T2I models then draw the whole courtyard/building.  This helper is generic:
    it cuts at common relational markers and keeps the core component phrase.
    """
    explicit = part.get("part_description_core") or part.get("t2i_subject")
    if explicit not in (None, "", []):
        return normalize_prompt_text(str(explicit).replace("_", " "))

    description = component_source_description(part)
    lowered = " {} ".format(description.lower())
    split_index = None
    for marker in RELATIONAL_SPLIT_MARKERS:
        index = lowered.find(marker)
        if index <= 0:
            continue
        candidate = description[:index].strip(" ,;:|.-")
        if len(candidate.split()) >= 2:
            split_index = index if split_index is None else min(split_index, index)

    if split_index is None:
        return normalize_prompt_text(description)
    core = normalize_prompt_text(description[:split_index].strip(" ,;:|.-"))
    return core or normalize_prompt_text(description)


def _drop_directional_prefix(text):
    words = normalize_prompt_text(text).split()
    while words and words[0].lower().strip(" ,;:-_") in DIRECTIONAL_PREFIX_WORDS:
        words = words[1:]
    return " ".join(words) or normalize_prompt_text(text)


def component_visual_subject(part):
    """Return a concise T2I subject optimized for a single visible component.

    This is not a closed class map.  It removes layout-only direction words and
    rewrites ambiguous architectural planning nouns (bar/mass/void/plane) into
    visual object phrases so T2I does not create signage, shops, or whole scenes.
    """
    subject = _drop_directional_prefix(component_core_description(part))
    lowered = subject.lower()
    role = normalize_prompt_text(part.get("semantic_role", "")).lower()
    legacy = normalize_prompt_text(part.get("legacy_compatibility_hint", "")).lower()

    if "gallery bar" in lowered:
        return "long rectangular gallery wing architectural volume with detailed stone facade"
    if lowered.endswith(" bar") or " bar " in lowered:
        prefix = normalize_prompt_text(re.sub(r"\bbar\b", "", subject)).strip()
        return normalize_prompt_text(
            "long rectangular {} architectural wing volume".format(prefix)
            if prefix else
            "long rectangular architectural wing volume"
        )
    if "void" in lowered:
        return normalize_prompt_text(
            subject.replace("void marker", "open architectural frame")
            .replace("void", "open framed void")
        )
    if "paving plane" in lowered or lowered.endswith(" plane"):
        return normalize_prompt_text(subject.replace("paving plane", "flat stone paving slab").replace(" plane", " slab"))
    if "mass" in lowered or role == "primary mass" or legacy == "mass":
        if "wall" in lowered:
            return normalize_prompt_text(subject.replace("wall mass", "thick exterior wall segment"))
        return normalize_prompt_text(subject + " architectural massing volume with facade detail")
    return subject


def component_context_tail(part):
    description = component_source_description(part)
    core = component_core_description(part)
    if description.lower().startswith(core.lower()):
        tail = description[len(core):].strip(" ,;:|.-")
        return normalize_prompt_text(tail)
    return ""


def _context_warning_text(part):
    source = component_source_description(part)
    core = component_core_description(part)
    tail = component_context_tail(part)
    spatial_relations = part.get("spatial_relations") or []
    if isinstance(spatial_relations, (str, bytes)):
        spatial_relations = [str(spatial_relations)]
    relation_text = "; ".join(str(item) for item in spatial_relations if str(item).strip())
    payload = " ".join([source, tail, relation_text]).lower()
    if not tail and not relation_text and not any(word in payload for word in CONTEXT_WORDS):
        return ""
    return normalize_prompt_text(
        "ignore placement/context words from the layout; draw only the visual subject '{}'; "
        "do not draw the surrounding building, courtyard, adjacent wings, scene, or neighboring parts".format(core)
    )


def _semantic_hint_text(part):
    hints = []
    role = normalize_prompt_text(part.get("semantic_role", ""))
    detail = normalize_prompt_text(part.get("detail_level", ""))
    material = normalize_prompt_text(part.get("material_hint", ""))
    if role:
        hints.append("semantic role for this single part: {}".format(role))
    if detail:
        hints.append("detail level: {}".format(detail))
    if material:
        hints.append("material hint: {}".format(material))
    return "; ".join(hints)


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


def front_axes_for_part(target_class, size):
    """Choose silhouette axes from bbox geometry, not part-specific classes.

    V7.2 intentionally avoids class-specific axis rules.  The reference image
    should show the two largest layout extents, because those extents describe
    the visible silhouette the T2I model must respect for arbitrary component
    text.  Wall-like placeholders remain the only semantic special case in the
    caller, where they can skip T2I entirely.
    """
    values = [float(value) for value in size]
    axis_names = ("x", "y", "z")
    ranked = sorted(range(3), key=lambda index: values[index], reverse=True)
    chosen = sorted(ranked[:2])
    return tuple(axis_names[index] for index in chosen)


def visual_ratio_for_part(target_class, size):
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    front_axes = front_axes_for_part(target_class, size)
    values = [float(value) for value in size]
    width = max(values[axis_to_index[front_axes[0]]], 1e-6)
    height = max(values[axis_to_index[front_axes[1]]], 1e-6)
    return width / height, front_axes, width, height


def ratio_bucket_phrase(ratio):
    if ratio >= 3.2:
        return "extremely wide and short"
    if ratio >= 2.0:
        return "very wide and low"
    if ratio >= 1.35:
        return "wide horizontal"
    if ratio <= 0.32:
        return "extremely tall and narrow"
    if ratio <= 0.55:
        return "very tall and narrow"
    if ratio <= 0.78:
        return "tall vertical"
    return "near-square"


def build_ratio_condition_text(target_class, size):
    ratio, front_axes, width, height = visual_ratio_for_part(target_class, size)
    ratio_text = "{:.2f}:1".format(ratio)
    phrase = ratio_bucket_phrase(ratio)
    axis_text = "{} by {}".format(front_axes[0].upper(), front_axes[1].upper())
    subject = "main visible silhouette"
    return normalize_prompt_text(
        "{} must match layout {} ratio {}, {} ({:.4f} by {:.4f} layout units); "
        "compose the object to fill this proportion, do not make it generic square".format(
            subject,
            axis_text,
            ratio_text,
            phrase,
            width,
            height,
        )
    )


def is_wall_class(target_class):
    return str(target_class).lower().startswith("wall")


def build_wall_facade_prompt(part):
    target_text = part.get("target_prompt") or part.get("source_actor_label") or part.get("target_class", "wall")
    target_text = normalize_prompt_text(str(target_text).replace("_", " "))
    return normalize_prompt_text(
        "{}; {}; semantic hint from layout label: {}; {}; {}".format(
            WALL_FACADE_PROMPT_TEMPLATE,
            FACADE_TEXTURE_FRAME_TEXT,
            target_text,
            GLOBAL_CIVIC_STYLE_TEXT,
            WALL_TEXTURE_POLICY_TEXT,
        )
    )


def negative_prompt_for_part(part):
    if is_wall_class(part.get("target_class", "object")):
        return WALL_FACADE_NEGATIVE_PROMPT
    return NEGATIVE_PROMPT


def build_part_prompt(part):
    target_class = part.get("target_class", "object")
    if is_wall_class(target_class):
        return build_wall_facade_prompt(part)
    target_text = component_visual_subject(part)
    base = (
        "single isolated standalone architectural component asset: {}, exactly one detached physical object, "
        "orthographic front product render, plain white background, no text in the image, "
        "no labels, no signboard, no readable letters, "
        "not an entire building, not a courtyard, not a scene, not a facade crop, "
        "not mounted on another surface"
    ).format(target_text)
    context_warning = _context_warning_text({**part, "part_description_core": target_text})
    context_text = ", {}".format(context_warning) if context_warning else ""
    semantic_hint = _semantic_hint_text(part)
    semantic_text = ", {}".format(semantic_hint) if semantic_hint else ""
    size = part.get("actor_size", part.get("bbox", {}).get("size", []))
    shape_phrase = _shape_phrase(size)
    shape_text = ", {}".format(shape_phrase) if shape_phrase else ""
    ratio_condition = build_ratio_condition_text(target_class, size) if len(size) == 3 else ""
    ratio_text = ", {}".format(ratio_condition) if ratio_condition else ""
    return normalize_prompt_text(
        "{}, open-vocabulary architectural component, isolated single component, {}, {}, {}, {},"
        " no background scene, no ground plane, no surrounding parts, not a composite object,"
        " studio product render, only one object, {}{}{}{}".format(
            base,
            ORTHOGRAPHIC_ASSET_TEXT,
            GLOBAL_CIVIC_STYLE_TEXT,
            BACKGROUND_CONTROL_TEXT,
            GENERIC_OBJECT_ONLY_TEXT,
            REFERENCE_FRAME_TEXT,
            semantic_text,
            shape_text,
            ratio_text,
        )
    )


def recommended_t2i_canvas_size(target_class, size):
    """Return width/height for ratio-aware T2I requests.

    SiliconFlow maps these dimensions to the nearest supported image_size, but
    passing aspect-aware dimensions still lets the provider pick a non-square
    canvas before generation.
    """
    ratio, _, _, _ = visual_ratio_for_part(target_class, size)
    if ratio >= 1.45:
        return 1664, 928
    if ratio <= 0.70:
        return 928, 1664
    if ratio >= 1.20:
        return 1472, 1140
    if ratio <= 0.84:
        return 1140, 1472
    return 1328, 1328


def prompt_hash(prompt, negative_prompt=NEGATIVE_PROMPT):
    payload = "{}\n---\n{}".format(
        normalize_prompt_text(prompt),
        normalize_prompt_text(negative_prompt),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
