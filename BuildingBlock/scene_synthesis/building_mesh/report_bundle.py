"""Static HTML report bundle for BuildingBlock-Hunyuan runs."""

import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from .visualization import (
    CLASS_COLORS,
    export_contract_part_assembly_glb,
    export_layout_3d_boxes_glb,
    render_contract_part_assembly,
    render_contract_part_assembly_views,
    render_layout_3d_boxes,
    render_layout_3d_box_views,
    render_layout_projection,
    render_layout_xy,
    render_obj_mesh,
)


def rel(path, base):
    if path is None:
        return None
    return os.path.relpath(str(path), str(base))


def load_json(path):
    return json.loads(Path(path).read_text())


def find_part_reference_image(part_id, contract):
    for part in contract["parts"]:
        if part["part_id"] == part_id:
            return part.get("reference_image_path") or part.get("reference_image")
    return None


def find_part_prompt(part_id, contract):
    for part in contract["parts"]:
        if part["part_id"] == part_id:
            return part.get("part_prompt") or part.get("target_prompt") or ""
    return ""


def copy_asset_to_report(source_path, dest_path):
    if not source_path:
        return None
    source_path = Path(source_path)
    if not source_path.exists():
        return None
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    return str(dest_path)


def copy_mesh_asset_to_report(source_path, dest_path):
    if not source_path:
        return None
    source_path = Path(source_path)
    if not source_path.exists():
        return None
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != dest_path.resolve():
        if dest_path.exists() or dest_path.is_symlink():
            dest_path.unlink()
        dest_path.symlink_to(source_path.resolve())
    return str(dest_path)


def _existing_path(path):
    if not path:
        return None
    path = Path(path)
    if path.exists():
        return str(path)
    return None


def _infer_part_output_paths(run_dir, part_id, manifest_part):
    part_dir = Path(run_dir) / "parts" / part_id
    raw_output_path = _existing_path(manifest_part.get("raw_output_path"))
    if raw_output_path is None:
        raw_output_path = _existing_path(part_dir / f"{part_id}.obj")

    normalized_output_path = _existing_path(manifest_part.get("normalized_output_path"))
    if normalized_output_path is None:
        normalized_output_path = raw_output_path

    glb_path = None
    if raw_output_path is not None:
        glb_path = _existing_path(Path(raw_output_path).with_suffix(".glb"))
    if glb_path is None:
        glb_path = _existing_path(part_dir / f"{part_id}.glb")

    placeholder_output_path = _existing_path(manifest_part.get("placeholder_output_path"))
    if placeholder_output_path is None:
        placeholder_output_path = _existing_path(
            Path(run_dir) / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj"
        )

    return {
        "raw_output_path": raw_output_path,
        "normalized_output_path": normalized_output_path,
        "glb_path": glb_path,
        "placeholder_output_path": placeholder_output_path,
    }


def _failure_payload_for_report(failures, rows):
    failure_records = list(failures.get("failures", []))
    recorded_part_ids = {
        record.get("part_id")
        for record in failure_records
        if isinstance(record, dict)
    }
    for row in rows:
        if row["status"] != "failed" or row["part_id"] in recorded_part_ids:
            continue
        failure_records.append({
            "part_id": row["part_id"],
            "target_class": row["target_class"],
            "failure_type": "missing_output_mesh",
            "raw_output_path": row["raw_output_path"],
            "normalized_output_path": row["normalized_output_path"],
            "placeholder_output_path": row["placeholder_output_path"],
            "lifecycle_states": row["lifecycle_states"],
        })

    return {
        **failures,
        "failures": failure_records,
    }


def build_report(run_dir):
    run_dir = Path(run_dir).resolve()
    report_dir = run_dir / "report"
    assets_dir = report_dir / "assets"
    parts_assets_dir = assets_dir / "parts"
    report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    parts_assets_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(run_dir / "manifest.json")
    failures = load_json(run_dir / "failures.json")
    contract_path = Path(manifest["parts"][0]["contract_path"])
    contract = load_json(contract_path)

    raw_assembly_path = _existing_path(manifest.get("raw_assembly_path"))
    if raw_assembly_path is None:
        raw_assembly_path = _existing_path(run_dir / "assemblies" / "raw_hunyuan_assembly.obj")

    placeholder_assembly_path = _existing_path(manifest.get("placeholder_assembly_path"))
    if placeholder_assembly_path is None:
        placeholder_assembly_path = _existing_path(
            run_dir / "assemblies" / "placeholder_filled_assembly.obj"
        )

    layout_overview = render_layout_xy(contract, assets_dir / "layout_overview.png")
    layout_xz = render_layout_projection(contract, assets_dir / "layout_xz.png", axes=("x", "z"))
    layout_yz = render_layout_projection(contract, assets_dir / "layout_yz.png", axes=("y", "z"))
    layout_3d = render_layout_3d_boxes(
        contract,
        assets_dir / "layout_3d_boxes.png",
        figsize=(8, 5.5),
        title="3D layout bbox overview",
    )
    layout_3d_views = render_layout_3d_box_views(
        contract,
        assets_dir,
        "layout_3d_view",
        title="3D layout bbox",
    )
    try:
        layout_3d_model = export_layout_3d_boxes_glb(contract, assets_dir / "layout_3d_boxes.glb")
    except Exception:
        layout_3d_model = None
    try:
        assembly_raw_model = export_contract_part_assembly_glb(
            contract,
            run_dir,
            assets_dir / "assembly_raw_colored.glb",
        )
    except Exception:
        assembly_raw_model = None
    raw_mesh_views = render_contract_part_assembly_views(
        contract,
        run_dir,
        assets_dir,
        "assembly_raw_view",
        title="Raw assembled mesh (per-part colors)",
        max_faces_per_mesh=9000,
    )
    placeholder_mesh_views = raw_mesh_views
    raw_render = raw_mesh_views.get("iso")
    placeholder_render = raw_render
    raw_assembly_asset = copy_mesh_asset_to_report(
        raw_assembly_path,
        assets_dir / "raw_hunyuan_assembly.obj",
    )
    placeholder_assembly_asset = copy_mesh_asset_to_report(
        placeholder_assembly_path,
        assets_dir / "placeholder_filled_assembly.obj",
    )

    rows = []
    for part in manifest["parts"]:
        part_id = part["part_id"]
        target_class = part.get("target_class", "object")
        inferred = _infer_part_output_paths(run_dir, part_id, part)
        raw_output_path = inferred["raw_output_path"]
        normalized_output_path = inferred["normalized_output_path"]
        placeholder_output_path = inferred["placeholder_output_path"]
        display_mesh_path = normalized_output_path or placeholder_output_path
        mesh_caption = "normalized mesh" if normalized_output_path else "placeholder mesh"
        mesh_render = None
        if display_mesh_path:
            mesh_render = render_obj_mesh(
                display_mesh_path,
                parts_assets_dir / f"{part_id}_mesh.png",
                title=target_class,
                color=CLASS_COLORS.get(target_class, CLASS_COLORS["object"]),
                max_faces=22000,
                figsize=(5, 4),
            )
        layout_render = render_layout_xy(
            contract,
            parts_assets_dir / f"{part_id}_layout.png",
            highlight_part_id=part_id,
        )
        reference_image = find_part_reference_image(part_id, contract)
        report_reference_image = copy_asset_to_report(
            reference_image,
            parts_assets_dir / f"{part_id}_reference.png",
        )
        raw_glb = copy_asset_to_report(
            inferred["glb_path"],
            parts_assets_dir / f"{part_id}.glb",
        )
        prompt = find_part_prompt(part_id, contract)
        status = "success" if raw_output_path else "failed"
        rows.append({
            "part_id": part_id,
            "target_class": target_class,
            "status": status,
            "prompt": prompt,
            "reference_image": report_reference_image,
            "model_glb": raw_glb,
            "mesh_render": mesh_render,
            "layout_render": layout_render,
            "raw_output_path": raw_output_path,
            "normalized_output_path": normalized_output_path,
            "placeholder_output_path": placeholder_output_path,
            "display_mesh_path": display_mesh_path,
            "mesh_caption": mesh_caption if display_mesh_path else "mesh",
            "lifecycle_states": part.get("lifecycle_states", []),
        })

    class_distribution = Counter(row["target_class"] for row in rows)
    failed_rows = [row for row in rows if row["status"] == "failed"]
    report_failures = _failure_payload_for_report(failures, rows)
    summary = {
        "run_dir": str(run_dir),
        "layout_id": manifest.get("layout_id"),
        "num_parts": len(rows),
        "num_success": sum(row["status"] == "success" for row in rows),
        "num_failures": len(failed_rows),
        "num_failure_records": len(report_failures.get("failures", [])),
        "class_distribution": dict(sorted(class_distribution.items())),
        "layout_overview": rel(layout_overview, report_dir),
        "layout_xz": rel(layout_xz, report_dir),
        "layout_yz": rel(layout_yz, report_dir),
        "layout_3d": rel(layout_3d, report_dir),
        "layout_3d_views": {
            name: rel(path, report_dir) for name, path in layout_3d_views.items()
        },
        "layout_3d_model": rel(layout_3d_model, report_dir) if layout_3d_model else None,
        "assembly_raw_image": rel(raw_render, report_dir) if raw_render else None,
        "assembly_placeholder_image": rel(placeholder_render, report_dir) if placeholder_render else None,
        "assembly_raw_views": {
            name: rel(path, report_dir) for name, path in raw_mesh_views.items()
        },
        "assembly_raw_model": rel(assembly_raw_model, report_dir) if assembly_raw_model else None,
        "assembly_placeholder_views": {
            name: rel(path, report_dir) for name, path in placeholder_mesh_views.items()
        },
        "assembly_raw_obj": rel(raw_assembly_asset, report_dir) if raw_assembly_asset else None,
        "assembly_placeholder_obj": (
            rel(placeholder_assembly_asset, report_dir)
            if placeholder_assembly_asset else None
        ),
    }
    parts_json = []
    for row in rows:
        parts_json.append({
            **row,
            "reference_image": rel(row["reference_image"], report_dir),
            "model_glb": rel(row["model_glb"], report_dir) if row["model_glb"] else None,
            "mesh_render": rel(row["mesh_render"], report_dir) if row["mesh_render"] else None,
            "layout_render": rel(row["layout_render"], report_dir) if row["layout_render"] else None,
        })

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (report_dir / "parts.json").write_text(json.dumps(parts_json, indent=2, sort_keys=True) + "\n")
    (report_dir / "index.html").write_text(render_html(summary, parts_json, report_failures))
    return report_dir


def render_html(summary, parts, failures):
    def image_panel(title, image_path):
        return f"""
        <div class="panel">
          <h2>{html.escape(title)}</h2>
          <img src="{html.escape(image_path or '')}">
        </div>
        """

    def model_or_image_panel(title, model_path=None, image_path=None, *, exposure="1.2", shadow_intensity="0.8"):
        escaped_title = html.escape(title)
        if model_path:
            return f"""
        <div class="panel">
          <h2>{escaped_title}</h2>
          <model-viewer src="{html.escape(model_path)}" camera-controls auto-rotate interaction-prompt="none" exposure="{html.escape(exposure)}" shadow-intensity="{html.escape(shadow_intensity)}"></model-viewer>
        </div>
        """
        return image_panel(title, image_path)

    def view_gallery(title, views):
        if not views:
            return ""
        order = ["iso", "front", "side", "top"]
        figures = []
        for name in order:
            if name not in views:
                continue
            figures.append(
                f"""<figure><img src="{html.escape(views[name] or '')}"><figcaption>{html.escape(name)}</figcaption></figure>"""
            )
        return f"""
        <section class="panel wide">
          <h2>{html.escape(title)}</h2>
          <div class="view-grid">{''.join(figures)}</div>
        </section>
        """

    cards = []
    for part in parts:
        part_viewer = f"""
          <div class="part-viewer-wrap">
            <model-viewer
              src="{html.escape(part.get('model_glb') or '')}"
              camera-controls
              auto-rotate
              interaction-prompt="none"
              exposure="1.25"
              shadow-intensity="1"
              shadow-softness="0.6"
              environment-image="neutral"
            ></model-viewer>
          </div>
        """ if part.get("model_glb") else f"""
          <div class="part-viewer-wrap">
            <img src="{html.escape(part.get('mesh_render') or '')}" alt="{html.escape(part['part_id'])} mesh preview">
          </div>
        """
        cards.append(f"""
        <section class="card {html.escape(part['status'])}">
          <div class="meta">
            <h3>{html.escape(part['target_class'])}</h3>
            <code>{html.escape(part['part_id'])}</code>
            <p class="status">{html.escape(part['status'])}</p>
          </div>
          <p class="prompt">{html.escape(part['prompt'])}</p>
          <div class="imgs">
            <figure><img src="{html.escape(part['reference_image'] or '')}"><figcaption>reference image</figcaption></figure>
            <figure><img src="{html.escape(part['mesh_render'] or '')}"><figcaption>{html.escape(part.get('mesh_caption') or 'mesh')}</figcaption></figure>
            <figure><img src="{html.escape(part['layout_render'] or '')}"><figcaption>layout highlight</figcaption></figure>
          </div>
          {part_viewer}
        </section>
        """)
    failures_html = "<p>No failures.</p>" if not failures.get("failures") else "<pre>{}</pre>".format(
        html.escape(json.dumps(failures, indent=2))
    )
    class_dist = ", ".join(f"{k}: {v}" for k, v in summary["class_distribution"].items())
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>BuildingBlock-Hunyuan Run Report</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f7f7f5; color: #222; }}
    h1, h2 {{ margin-bottom: 0.3rem; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 18px 0; }}
    .panel, .card {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .panel img {{ width: 100%; height: 230px; object-fit: contain; background: #fff; }}
    .panel model-viewer {{ height: 230px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(760px, 1fr)); gap: 16px; }}
    .card {{ border-left: 6px solid #999; }}
    .card.success {{ border-left-color: #54A24B; }}
    .card.failed {{ border-left-color: #E45756; }}
    .meta {{ display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
    code {{ font-size: 11px; color: #555; }}
    .status {{ font-weight: 700; text-transform: uppercase; color: #555; }}
    .prompt {{ font-size: 13px; line-height: 1.45; background: #fafafa; padding: 10px 12px; border-radius: 8px; max-width: none; }}
    .imgs {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; height: 220px; object-fit: contain; background: #fff; border: 1px solid #eee; border-radius: 8px; }}
    figcaption {{ font-size: 12px; color: #666; text-align: center; }}
    .part-viewer-wrap {{
      width: calc((100% - 20px) / 3);
      min-width: 220px;
      max-width: 100%;
      margin-top: 10px;
    }}
    .part-viewer-wrap img,
    .part-viewer-wrap model-viewer {{
      width: 100%;
      height: 220px;
      margin-top: 0;
      object-fit: contain;
      border: 1px solid #eee;
      border-radius: 8px;
      background-color: #151922;
    }}
    model-viewer {{
      width: 100%;
      height: 430px;
      margin-top: 12px;
      border: 1px solid #d8d8d8;
      border-radius: 10px;
      background-color: #151922;
      background-image:
        linear-gradient(45deg, #222836 25%, transparent 25%),
        linear-gradient(-45deg, #222836 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #222836 75%),
        linear-gradient(-45deg, transparent 75%, #222836 75%);
      background-size: 24px 24px;
      background-position: 0 0, 0 12px, 12px -12px, -12px 0px;
    }}
    .hero {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; margin: 20px 0; }}
    .hero .panel img {{ height: 420px; }}
    .wide {{ grid-column: 1 / -1; }}
    .view-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
    .view-grid img {{ width: 100%; height: 230px; object-fit: contain; background: #101318; border: 1px solid #222; border-radius: 8px; }}
    .downloads {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 18px; }}
    .downloads a {{ background: #222; color: #fff; text-decoration: none; padding: 8px 12px; border-radius: 8px; font-size: 13px; }}
    @media (max-width: 840px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .part-viewer-wrap {{ width: 100%; min-width: 0; }}
    }}
  </style>
</head>
<body>
  <h1>BuildingBlock-Hunyuan Run Report</h1>
  <p><b>Run:</b> {html.escape(summary['run_dir'])}</p>
  <p><b>Parts:</b> {summary['num_parts']} | <b>Success:</b> {summary['num_success']} | <b>Failures:</b> {summary['num_failures']} | <b>Failure records:</b> {summary.get('num_failure_records', summary['num_failures'])}</p>
  <p><b>Class distribution:</b> {html.escape(class_dist)}</p>
  <div class="downloads">
    <a href="{html.escape(summary.get('assembly_raw_obj') or '#')}">Download raw assembled OBJ</a>
    <a href="{html.escape(summary.get('assembly_placeholder_obj') or '#')}">Download complete assembly OBJ</a>
  </div>
  <h2>Overall 3D layout + assembled mesh</h2>
  <div class="hero">
    {image_panel('Large 3D layout boxes', summary.get('layout_3d'))}
    {model_or_image_panel('Large assembled mesh', summary.get('assembly_raw_model'), summary.get('assembly_raw_image') or summary.get('assembly_placeholder_image'))}
  </div>
  <div class="summary">
    {view_gallery('Layout multi-view renders', summary.get('layout_3d_views', {}))}
    {view_gallery('Assembled mesh multi-view renders', summary.get('assembly_raw_views', {}) or summary.get('assembly_placeholder_views', {}))}
  </div>
  <div class="summary">
    <div class="panel"><h2>Layout XY top</h2><img src="{html.escape(summary['layout_overview'])}"></div>
    <div class="panel"><h2>Layout XZ elevation</h2><img src="{html.escape(summary['layout_xz'])}"></div>
    <div class="panel"><h2>Layout YZ elevation</h2><img src="{html.escape(summary['layout_yz'])}"></div>
    {model_or_image_panel('Layout 3D bbox', summary.get('layout_3d_model'), summary.get('layout_3d'), exposure='1.2', shadow_intensity='0.7')}
    {model_or_image_panel('Raw assembly', summary.get('assembly_raw_model'), summary.get('assembly_raw_image') or summary.get('assembly_placeholder_image'))}
    <div class="panel"><h2>Placeholder assembly</h2><img src="{html.escape(summary['assembly_placeholder_image'] or '')}"></div>
  </div>
  <h2>Failures</h2>
  <div class="panel">{failures_html}</div>
  <h2>Part Board</h2>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
