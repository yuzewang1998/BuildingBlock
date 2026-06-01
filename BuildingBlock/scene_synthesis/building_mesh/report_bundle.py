"""Static HTML report bundle for BuildingBlock-Hunyuan runs."""

import html
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .visualization import (
    CLASS_COLORS,
    S2_VLM_SIX_VIEWS,
    compose_image_grid,
    export_contract_part_assembly_glb,
    export_layout_3d_boxes_glb,
    export_obj_mesh_glb,
    layout_vertex_to_glb_viewer,
    part_color,
    render_contract_part_assembly,
    render_contract_part_assembly_views,
    render_layout_3d_boxes,
    render_layout_3d_box_views,
    render_layout_projection,
    render_layout_text_callouts,
    render_layout_xy,
    render_obj_mesh,
    render_obj_mesh_strict,
    render_obj_mesh_views,
    render_s1_style_layout_multiview,
)


REPORT_ASSET_VERSION = "v6_visual_fix_5_s1_layout_focus"
MODEL_VIEWER_SOURCE = "https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"
MODEL_VIEWER_LOCAL_ASSET = "assets/model-viewer.min.js"
S2_MESH_RENDER_NOTE = "bright filled mesh surface preview (not point cloud scatter)"
QUALITY_SECTION_START = "<!-- ARCHSTUDIO_S2_QUALITY_AGENT_START -->"
QUALITY_SECTION_END = "<!-- ARCHSTUDIO_S2_QUALITY_AGENT_END -->"
S2_TRACE_SECTION_START = "<!-- ARCHSTUDIO_S2_TRACE_AGENT_START -->"
S2_TRACE_SECTION_END = "<!-- ARCHSTUDIO_S2_TRACE_AGENT_END -->"
S2_REPAIR_SECTION_START = "<!-- ARCHSTUDIO_S2_REPAIR_AGENT_START -->"
S2_REPAIR_SECTION_END = "<!-- ARCHSTUDIO_S2_REPAIR_AGENT_END -->"
FOCUSED_CRITIC_SECTION_START = "<!-- ARCHSTUDIO_S2_FOCUSED_CRITICS_START -->"
FOCUSED_CRITIC_SECTION_END = "<!-- ARCHSTUDIO_S2_FOCUSED_CRITICS_END -->"


def _report_display_image_for_dark_render(source_path, report_root, rel_path=None, darkness_threshold=55.0, *, enabled=True):
    """Return a report-relative image path that is readable on the HTML page.

    Some mesh/assembly renders intentionally use a nearly black renderer
    background.  They are valid evidence files, but in a dark-card report they
    can look like fully black thumbnails.  For *display only*, create a sibling
    enhanced copy that lifts near-black background pixels to a light neutral
    color while preserving the original image and JSON request paths.

    Returns:
        (relative_path, enhanced, mean_luminance)
    """
    if not source_path:
        return str(rel_path or ""), False, None
    if not enabled:
        return str(rel_path or source_path), False, None
    try:
        source_path = Path(source_path)
        report_root = Path(report_root)
        if not source_path.is_absolute():
            source_path = report_root / source_path
        if rel_path is None:
            rel_path = os.path.relpath(str(source_path), str(report_root))
        else:
            rel_path = str(rel_path)
        if not source_path.exists() or source_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return rel_path, False, None

        from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
        import numpy as np

        image = Image.open(source_path).convert("RGB")
        mean_luminance = float(sum(ImageStat.Stat(image).mean) / 3.0)
        pixels = np.array(image)
        dark_mask = pixels.max(axis=2) < 38
        dark_ratio = float(dark_mask.mean()) if dark_mask.size else 0.0
        # Boards can have bright mesh panels plus black letterbox margins; mean
        # luminance alone misses those.  Lift dark margins for display while
        # keeping the original linked for exact VLM-input auditability.
        if mean_luminance >= darkness_threshold and dark_ratio < 0.08:
            return rel_path, False, mean_luminance

        dest_rel = Path("assets") / "display_enhanced" / Path(rel_path)
        dest = report_root / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_mtime < source_path.stat().st_mtime:
            # Renderer background / board letterbox margins are usually almost
            # black.  Lift only those pixels so colored geometry remains true to
            # the evidence image.
            foreground_mask = ~dark_mask
            output = pixels.copy()
            output[dark_mask] = np.array([232, 236, 242], dtype=output.dtype)
            # Sparse mesh renders can be only one-pixel colored samples.  Add a
            # very small display-only halo so users can inspect the shape in
            # the web report without changing the original evidence file.
            halo = np.array(
                Image.fromarray((foreground_mask.astype("uint8") * 255)).filter(ImageFilter.MaxFilter(3))
            ) > 0
            halo_only = halo & dark_mask
            output[halo_only] = np.array([80, 112, 170], dtype=output.dtype)
            foreground = foreground_mask & (pixels.max(axis=2) < 245)
            output[foreground] = np.maximum(output[foreground], np.array([34, 62, 118], dtype=output.dtype))
            enhanced = Image.fromarray(output)
            # If only board letterbox margins were dark, avoid autocontrast: it
            # can make already-readable light mesh surfaces look darker.
            if mean_luminance < darkness_threshold:
                enhanced = ImageOps.autocontrast(enhanced, cutoff=0.2)
                enhanced = ImageEnhance.Contrast(enhanced).enhance(1.08)
            enhanced.save(dest)
        return os.path.relpath(str(dest), str(report_root)), True, mean_luminance
    except Exception:
        return str(rel_path or source_path), False, None


def _focused_scope_step(scope):
    scope = str(scope or "")
    if scope == "part_image":
        return "Step 1", "单部件 T2I 参考图审查", "先检查每个 part 的文生图：是不是独立部件、是否符合文本和 bbox 比例、是否误生成整栋建筑或背景平面。不合格时应先重生图，再进入几何。"
    if scope == "part_mesh":
        return "Step 2", "单部件 3D mesh 审查", "在 T2I 通过后检查单个 part 的真 mesh 渲染：是否沿重力方向、是否适合 reference image 和 layout bbox。"
    if scope == "scene_assembly":
        return "Step A", "整体拼装 mesh 审查", "Assemble Geometry Agent：只看 assembled mesh 多视角真 mesh surface 渲染，检查缝隙、漂浮、穿插、重叠、比例/方向异常，并把建议具体路由到某个 part 的 snap / rerun I3D / rerun T2I+I3D。"
    return "Step ?", "其他 focused critic", "该 critic 使用独立 scope，不与其他证据混在同一次 VLM 判断里。"


def rel(path, base):
    if path is None:
        return None
    return os.path.relpath(str(path), str(base))


def load_json(path):
    return json.loads(Path(path).read_text())


def find_quality_summary(run_dir):
    candidates = [
        Path(run_dir) / "quality" / "v9_agent" / "quality_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_json(candidate)
    return None


def find_s2_agent_trace(run_dir):
    candidate = Path(run_dir) / "agent_trace" / "s2_agent_trace.json"
    if candidate.exists():
        try:
            return load_json(candidate), candidate
        except Exception:
            return None, candidate
    return None, candidate


def find_s2_repair_agent_trace(run_dir):
    candidate = Path(run_dir) / "agent_loop" / "agent_trace.json"
    if candidate.exists():
        try:
            return load_json(candidate), candidate
        except Exception:
            return None, candidate
    return None, candidate


def find_focused_visual_critics(run_dir):
    """Find focused VLM summaries from both legacy and per-iteration agent paths."""
    run_dir = Path(run_dir)
    legacy = run_dir / "agent_loop" / "focused_visual_critics" / "focused_visual_critics_summary.json"
    candidates = []
    if legacy.exists():
        candidates.append(legacy)
    candidates.extend(sorted((run_dir / "agent_loop" / "focused_vlm_interactions").glob("iteration_*/focused_visual_critics_summary.json")))
    loaded = []
    for candidate in candidates:
        try:
            payload = load_json(candidate)
            if isinstance(payload, dict):
                loaded.append((candidate, payload))
        except Exception:
            continue
    if not loaded:
        return None, legacy
    if len(loaded) == 1:
        return loaded[0][1], loaded[0][0]
    merged_results = []
    selected_part_ids = []
    scopes = []
    for candidate, payload in loaded:
        iteration_name = candidate.parent.name if candidate.parent.name.startswith("iteration_") else "legacy"
        for result in payload.get("results", []) or []:
            if isinstance(result, dict):
                merged_results.append({**result, "iteration_name": iteration_name, "focused_summary_path": str(candidate)})
        for part_id in payload.get("selected_part_ids", []) or []:
            if part_id not in selected_part_ids:
                selected_part_ids.append(part_id)
        for scope in payload.get("scopes", []) or []:
            if scope not in scopes:
                scopes.append(scope)
    first = loaded[0][1]
    merged = {
        **first,
        "schema_version": first.get("schema_version", "archstudio_s2.focused_visual_critics.v1"),
        "created_at": loaded[-1][1].get("created_at", first.get("created_at")),
        "provider": loaded[-1][1].get("provider", first.get("provider")),
        "model": loaded[-1][1].get("model", first.get("model")),
        "selected_part_ids": selected_part_ids,
        "scopes": scopes,
        "results": merged_results,
        "iteration_summaries": [str(candidate) for candidate, _ in loaded],
    }
    return merged, loaded[-1][0]


def refresh_quality_agent_report(run_dir):
    """Update the V9 quality section in an existing report without mesh rebuild.

    Full report generation exports/samples dense per-part meshes and is too slow
    for every agent QA iteration.  This fast path only reloads
    ``quality/v9_agent/quality_summary.json``, updates ``report/summary.json``,
    and replaces the marked QA section in ``report/index.html``.
    """
    run_dir = Path(run_dir).resolve()
    report_dir, summary_path, index_path = _ensure_report_scaffold(run_dir)

    summary = load_json(summary_path)
    # Copied experiment directories can carry an old report/summary.json.
    # Always bind refreshed 9999 sections to the current run so hook/status and
    # agent artifact links resolve under /projects/buildingblock/runs/<slug>/.
    summary["run_dir"] = str(run_dir)
    quality_summary = find_quality_summary(run_dir)
    if quality_summary:
        summary["quality_agent"] = quality_summary
    elif "quality_agent" in summary:
        summary.pop("quality_agent", None)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    html_text = index_path.read_text(encoding="utf-8", errors="ignore")
    section = render_quality_agent_section(summary, summary.get("quality_agent"))
    start = html_text.find(QUALITY_SECTION_START)
    end = html_text.find(QUALITY_SECTION_END)
    if start >= 0 and end > start:
        end += len(QUALITY_SECTION_END)
        html_text = html_text[:start] + section + html_text[end:]
    elif section:
        marker = '<div class="downloads">'
        index = html_text.find(marker)
        html_text = html_text[:index] + section + "\n" + html_text[index:] if index >= 0 else html_text + section
    html_text = ensure_s2_9999_agent_css(html_text)
    index_path.write_text(html_text, encoding="utf-8")
    return report_dir


def _ensure_report_scaffold(run_dir):
    """Return report/index paths, building a full report when the page is missing.

    The 9999 viewer serves ``report/index.html``.  Agent-only traces under
    ``agent_loop/`` are useful but invisible on 9999 until this scaffold exists,
    so refresh hooks call this helper before replacing marked sections.
    """
    run_dir = Path(run_dir).resolve()
    report_dir = run_dir / "report"
    summary_path = report_dir / "summary.json"
    index_path = report_dir / "index.html"
    if not summary_path.exists() or not index_path.exists():
        build_report(run_dir)
    return report_dir, summary_path, index_path


def write_s2_9999_refresh_hook(run_dir, *, trigger="manual", report_path=None):
    """Write a small hook manifest so the 9999 page has an explicit refresh signal."""
    run_dir = Path(run_dir).resolve()
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "s2_9999_refresh.json"
    payload = {
        "schema_version": "archstudio_s2_9999_refresh_hook.v1",
        "trigger": str(trigger),
        "run_dir": str(run_dir),
        "report_path": str(report_path or (report_dir / "index.html")),
        "agent_trace_path": str(run_dir / "agent_loop" / "agent_trace.json"),
        "agent_report_path": str(run_dir / "agent_loop" / "index.html"),
        "refreshed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


S2_9999_AGENT_CSS = """
<style id="s2-9999-agent-safe-css">
  html, body { max-width: 100%; overflow-x: hidden; }
  body { box-sizing: border-box; }
  *, *::before, *::after { box-sizing: border-box; }
  .panel, .card, .qa-section, .critic-qa-card, .agent-iteration-io { max-width: 100%; overflow-wrap: anywhere; }
  .qa-section { border: 1px solid #b7c7e8; background: #f8fbff; border-radius: 16px; padding: 16px; margin: 16px 0; }
  .qa-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 12px 0; }
  .qa-kpis div { background: white; border: 1px solid #d7e2f5; border-radius: 10px; padding: 10px; min-width: 0; }
  .qa-kpis b { display: block; color: #46628d; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
  .qa-kpis span { font-size: clamp(16px, 2.4vw, 22px); font-weight: 800; overflow-wrap: anywhere; }
  .downloads { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 18px; max-width: 100%; }
  .downloads a { background: #222; color: #fff; text-decoration: none; padding: 8px 12px; border-radius: 8px; font-size: 13px; max-width: 100%; overflow-wrap: anywhere; }
  .hint { color: #555; font-size: 13px; margin-top: 0; }
  .qa-table { width: 100%; max-width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 12px; table-layout: fixed; }
  .qa-table th, .qa-table td { border-bottom: 1px solid #d7e2f5; padding: 7px; text-align: left; vertical-align: top; overflow-wrap: anywhere; min-width: 0; }
  .qa-table th { color: #46628d; text-transform: uppercase; letter-spacing: .04em; }
  .critic-qa-card { background: #fffdf7; border: 1px solid #d6c4a2; border-radius: 16px; padding: 16px; margin: 14px 0; box-shadow: 0 8px 22px rgba(41, 36, 23, .07); overflow: hidden; }
  .critic-head { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; margin-bottom: 12px; max-width: 100%; }
  .critic-head h3 { margin: 4px 0 0; font-size: clamp(18px, 2.8vw, 24px); }
  .critic-score { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; min-width: 0; }
  .critic-score span { background: #223145; color: #fff; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; overflow-wrap: anywhere; }
  .critic-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(280px, .85fr); gap: 16px; align-items: start; }
  .critic-images, .focused-images { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(260px, 100%), 1fr)); gap: 12px; max-width: 100%; }
  .critic-board, .critic-tile, .agent-vlm-input-tile, figure { max-width: 100%; min-width: 0; }
  .critic-board img, .critic-tile img, .agent-vlm-input-tile img, figure img { max-width: 100%; object-fit: contain; }
  .critic-board img { width: 100%; max-height: 780px; background: #151922; border: 1px solid #222; border-radius: 12px; }
  .critic-tile img { width: 100%; height: min(260px, 45vw); background: #151922; border: 1px solid #222; border-radius: 10px; }
  .agent-refresh-status { background: #e9f7ef; border: 1px solid #9fd2b2; color: #214d32; border-radius: 10px; padding: 10px 12px; margin: 10px 0; font-size: 13px; overflow-wrap: anywhere; }
  .critic-error { background: #fff0f0; border: 1px solid #e7a3a3; color: #8a1f1f; border-radius: 10px; padding: 10px 12px; font-size: 12px; overflow-wrap: anywhere; }
  .agent-iteration-io { background: #fff; border: 2px solid #98b7e4; border-radius: 18px; padding: 16px; margin: 16px 0; overflow: hidden; }
  .agent-vlm-input-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(360px, 100%), 1fr)); gap: 14px; margin: 14px 0; max-width: 100%; }
  .agent-vlm-input-tile { margin: 0; background: #f8fbff; border: 1px solid #c4d5ef; border-radius: 12px; padding: 10px; overflow: hidden; }
  .agent-vlm-input-tile img { width: 100%; height: min(340px, 58vw); background: #edf2f8; border: 1px solid #a9b9cf; border-radius: 10px; }
  .agent-vlm-input-tile figcaption { color: #34465f; font-size: 12px; margin-top: 7px; text-align: left; overflow-wrap: anywhere; }
  .critic-prompt, .mini-json, pre { max-width: 100%; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
  .vlm-call-layout { display: grid; grid-template-columns: minmax(320px, 1.1fr) minmax(320px, .9fr); gap: 16px; align-items: start; margin: 12px 0; }
  .vlm-input-panel, .vlm-text-panel { min-width: 0; background: #fff; border: 1px solid #e3d4b9; border-radius: 14px; padding: 12px; }
  .vlm-input-panel h4, .vlm-text-panel h4 { margin: 0 0 8px; color: #385170; }
  .question-text { background: #eef5ff; border-left: 4px solid #4078c0; border-radius: 8px; padding: 10px; font-weight: 650; }
  .answer-summary { background: #eef9ef; border-left: 4px solid #3d8b50; border-radius: 8px; padding: 10px; }
  .focused-issues { padding-left: 20px; }
  .focused-issues li { margin: 8px 0; }
  .review-opinion { color: #23364f; }
  .repair-hint { color: #7a3d00; font-weight: 650; }
  .part-context-table { width: 100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 12px; table-layout: fixed; }
  .part-context-table th { width: 130px; color: #385170; text-align: left; vertical-align: top; padding: 6px; border-bottom: 1px solid #d7e2f5; }
  .part-context-table td { padding: 6px; border-bottom: 1px solid #d7e2f5; overflow-wrap: anywhere; }
  .verdict-banner { display:flex; justify-content:space-between; gap:10px; align-items:center; border-radius:12px; padding:10px 12px; margin:8px 0; }
  .verdict-accept { background:#1f7a3a !important; color:#fff !important; }
  .verdict-revise { background:#a84d00 !important; color:#fff !important; }
  .verdict-unknown { background:#53606f !important; color:#fff !important; }
  .part-repair-flow { margin-top: 12px; border: 1px solid #ecd6ae; background: #fffaf0; border-radius: 12px; padding: 10px; }
  .part-repair-flow ol { padding-left: 22px; margin: 8px 0; }
  .part-repair-flow li { margin: 10px 0; }
  .repair-generation-card { border: 2px solid #90b7df; background: #f7fbff; border-radius: 16px; padding: 14px; margin: 14px 0; overflow: hidden; }
  .repair-generation-card h3 { margin: 4px 0 10px; }
  .before-after-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(280px, 100%), 1fr)); gap: 12px; margin: 10px 0; }
  .before-after-grid figure { margin: 0; border: 1px solid #c8d7eb; border-radius: 12px; background: #fff; padding: 10px; }
  .before-after-grid img { width: 100%; height: min(300px, 50vw); object-fit: contain; background: #f7fafc; border-radius: 10px; }
  .prompt-compare { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(320px, 100%), 1fr)); gap: 12px; margin: 10px 0; }
  .prompt-compare pre { max-height: 220px; background: #f7fafc; border: 1px solid #d7e2f5; border-radius: 8px; padding: 8px; font-size: 11px; }
  .mini-downloads { margin: 6px 0; }
  .mini-downloads a { font-size: 11px; padding: 5px 8px; }
  .critic-prompt { max-height: 360px; background: #1e222b; color: #f4efe5; border-radius: 10px; padding: 12px; font-size: 11px; }
  .mini-json { max-height: 180px; background: #f4f7fb; border: 1px solid #d7e2f5; border-radius: 8px; padding: 8px; font-size: 11px; color: #203047; }
  .label { color: #6e5a34; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; overflow-wrap: anywhere; }
  @media (max-width: 900px) { .critic-grid, .vlm-call-layout { grid-template-columns: 1fr; } .critic-head { flex-direction: column; } .qa-table { display: block; overflow-x: auto; table-layout: auto; } }
</style>
"""


def ensure_s2_9999_agent_css(html_text):
    """Inject overflow-safe CSS for reports that predate the S2 agent section."""
    if 'id="s2-9999-agent-safe-css"' in html_text:
        return html_text
    head_index = html_text.lower().find("</head>")
    if head_index >= 0:
        return html_text[:head_index] + S2_9999_AGENT_CSS + html_text[head_index:]
    return S2_9999_AGENT_CSS + html_text


def refresh_s2_9999_report(run_dir, *, full=False, trigger="manual"):
    """Refresh the report served by the 9999 harness after an S2 agent event.

    ``full=True`` rebuilds all static report assets; otherwise this keeps the
    expensive mesh report intact and replaces only agent/quality marked sections.
    In both cases a hook manifest is touched so filesystem-based viewers can see
    that this experiment changed.
    """
    run_dir = Path(run_dir).resolve()
    if full:
        report_dir = build_report(run_dir)
        hook_path = write_s2_9999_refresh_hook(run_dir, trigger=trigger, report_path=report_dir / "index.html")
        # Re-render the agent section once after touching the hook manifest so
        # 9999 visibly shows when/how the page was refreshed.
        refresh_s2_repair_agent_report(run_dir)
    else:
        _ensure_report_scaffold(run_dir)
        hook_path = write_s2_9999_refresh_hook(run_dir, trigger=trigger, report_path=run_dir / "report" / "index.html")
        refresh_quality_agent_report(run_dir)
        report_dir = refresh_s2_repair_agent_report(run_dir)
    return {
        "report_dir": str(report_dir),
        "report_path": str(report_dir / "index.html"),
        "refresh_hook_path": str(hook_path),
        "full": bool(full),
        "trigger": str(trigger),
    }


def refresh_s2_repair_agent_report(run_dir):
    """Fast refresh only the S2 repair-agent section in report/index.html."""
    run_dir = Path(run_dir).resolve()
    report_dir, summary_path, index_path = _ensure_report_scaffold(run_dir)

    summary = load_json(summary_path)
    # Copied experiment directories can carry an old report/summary.json.
    # Always bind refreshed 9999 sections to the current run so hook/status and
    # agent artifact links resolve under /projects/buildingblock/runs/<slug>/.
    summary["run_dir"] = str(run_dir)
    repair_agent_trace, repair_trace_path = find_s2_repair_agent_trace(run_dir)
    if repair_agent_trace:
        summary["s2_repair_agent_trace"] = {
            **repair_agent_trace,
            "trace_path": rel(repair_trace_path, report_dir),
        }
    else:
        summary.pop("s2_repair_agent_trace", None)
    focused_critics, focused_critics_path = find_focused_visual_critics(run_dir)
    if focused_critics:
        summary["focused_visual_critics"] = {
            **focused_critics,
            "trace_path": rel(focused_critics_path, report_dir),
        }
    else:
        summary.pop("focused_visual_critics", None)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    html_text = index_path.read_text(encoding="utf-8", errors="ignore")
    focused_section = render_focused_visual_critics_section(summary, summary.get("focused_visual_critics"))
    focused_start = html_text.find(FOCUSED_CRITIC_SECTION_START)
    focused_end = html_text.find(FOCUSED_CRITIC_SECTION_END)
    if focused_start >= 0 and focused_end > focused_start:
        focused_end += len(FOCUSED_CRITIC_SECTION_END)
        html_text = html_text[:focused_start] + focused_section + html_text[focused_end:]
    elif focused_section:
        marker = S2_REPAIR_SECTION_START if S2_REPAIR_SECTION_START in html_text else QUALITY_SECTION_START
        index = html_text.find(marker)
        html_text = html_text[:index] + focused_section + "\n" + html_text[index:] if index >= 0 else html_text + focused_section
    section = render_s2_repair_agent_loop_section(summary, summary.get("s2_repair_agent_trace"))
    start = html_text.find(S2_REPAIR_SECTION_START)
    end = html_text.find(S2_REPAIR_SECTION_END)
    if start >= 0 and end > start:
        end += len(S2_REPAIR_SECTION_END)
        html_text = html_text[:start] + section + html_text[end:]
    elif section:
        marker = QUALITY_SECTION_START
        index = html_text.find(marker)
        html_text = html_text[:index] + section + "\n" + html_text[index:] if index >= 0 else html_text + section
    html_text = ensure_s2_9999_agent_css(html_text)
    index_path.write_text(html_text, encoding="utf-8")
    return report_dir


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


def find_part_t2i_metadata(part_id, contract):
    for part in contract["parts"]:
        if part["part_id"] != part_id:
            continue
        image_path = part.get("reference_image_path") or part.get("reference_image")
        if not image_path:
            return None
        metadata_path = Path(image_path).parent / "t2i_metadata.json"
        if metadata_path.exists():
            try:
                return load_json(metadata_path)
            except Exception:
                return None
    return None


def _safe_asset_name(value):
    text = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    text = "_".join(part for part in text.split("_") if part)
    return text[:160] or "asset"


def contract_part_by_id(contract):
    return {
        str(part.get("part_id")): part
        for part in contract.get("parts", [])
        if part.get("part_id")
    }


def display_label_for_part(contract_part, fallback_class):
    return (
        contract_part.get("open_vocab_label")
        or contract_part.get("part_description")
        or contract_part.get("source_part_name")
        or fallback_class
    )


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


def copy_optional_local_model_viewer(assets_dir):
    """Bundle model-viewer when a local downloaded copy is available.

    The report still falls back to the CDN when this file is absent, but copying
    it into assets makes remote/air-gapped inspection less fragile.
    """
    existing_asset = Path(assets_dir) / "model-viewer.min.js"
    if existing_asset.exists() and existing_asset.stat().st_size > 100000:
        return str(existing_asset)
    candidates = [
        Path("vendor/model-viewer.min.js"),
        Path("/tmp/model-viewer.min.js"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 100000:
            return copy_asset_to_report(candidate, Path(assets_dir) / "model-viewer.min.js")
    return None


def _existing_path(path):
    if not path:
        return None
    path = Path(path)
    if path.exists():
        return str(path)
    return None


def _infer_part_output_paths(run_dir, part_id, manifest_part):
    part_dir = Path(run_dir) / "parts" / part_id
    assembly_part_dir = Path(run_dir) / "assemblies" / "parts"
    placeholder_output_path = _existing_path(manifest_part.get("placeholder_output_path"))
    if placeholder_output_path is None:
        placeholder_output_path = _existing_path(
            Path(run_dir) / "assemblies" / "placeholders" / f"{part_id}__placeholder.obj"
        )
    raw_output_path = _existing_path(manifest_part.get("raw_output_path"))
    if raw_output_path is None:
        candidate_raw_path = _existing_path(part_dir / f"{part_id}.obj")
        if not (
            candidate_raw_path
            and placeholder_output_path
            and manifest_part.get("target_class") == "wall"
        ):
            raw_output_path = candidate_raw_path

    normalized_output_path = _existing_path(manifest_part.get("normalized_output_path"))
    if normalized_output_path is None:
        normalized_output_path = _existing_path(assembly_part_dir / f"{part_id}.obj")
    if normalized_output_path is None and not (
        placeholder_output_path and manifest_part.get("target_class") == "wall"
    ):
        normalized_output_path = raw_output_path

    glb_path = None
    if normalized_output_path is not None:
        glb_path = _existing_path(Path(normalized_output_path).with_suffix(".glb"))
    if glb_path is None:
        glb_path = _existing_path(assembly_part_dir / f"{part_id}.glb")
    if glb_path is None:
        glb_path = _existing_path(part_dir / f"{part_id}.glb")

    return {
        "raw_output_path": raw_output_path,
        "normalized_output_path": normalized_output_path,
        "glb_path": glb_path,
        "placeholder_output_path": placeholder_output_path,
    }


def _export_report_part_glb(display_mesh_path, destination_path, part, index):
    if not display_mesh_path:
        return None
    try:
        return export_obj_mesh_glb(
            display_mesh_path,
            destination_path,
            color=part_color(part, index=index),
            name=part.get("part_id"),
        )
    except Exception:
        return None


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
    copy_optional_local_model_viewer(assets_dir)

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
    layout_text_callouts = render_layout_text_callouts(
        contract,
        assets_dir / "layout_text_callouts_xz.png",
        axes=("x", "z"),
        title="Layout XZ part-to-text callouts",
    )
    layout_s1_style_multiview = render_s1_style_layout_multiview(
        contract,
        assets_dir / "layout_s1_style_visual_critic_multiview.png",
        title="ArchStudio-S2 layout · S1-style plain numeric labels",
    )
    layout_s1_style_label_coverage = (
        Path(layout_s1_style_multiview).with_name(
            "layout_s1_style_visual_critic_multiview_label_coverage.json"
        )
        if layout_s1_style_multiview
        else None
    )
    layout_3d = render_layout_3d_boxes(
        contract,
        assets_dir / "layout_3d_boxes.png",
        figsize=(8, 5.5),
        title="3D layout bbox overview (layout z-up frame)",
        transform=None,
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
            assets_dir / "assembly_raw_per_part_color_yup.glb",
            max_vertices_per_part=65000,
            include_placeholders=False,
        )
    except Exception:
        assembly_raw_model = None
    try:
        assembly_complete_model = export_contract_part_assembly_glb(
            contract,
            run_dir,
            assets_dir / "assembly_complete_per_part_color_yup.glb",
            max_vertices_per_part=65000,
            include_placeholders=True,
            placeholder_alpha=80,
        )
    except Exception:
        assembly_complete_model = None
    report_max_faces_per_mesh = int(os.environ.get("S2_REPORT_MESH_RENDER_MAX_FACES_PER_MESH", "8000"))
    raw_mesh_views = render_contract_part_assembly_views(
        contract,
        run_dir,
        assets_dir,
        "assembly_raw_view",
        title="Assembled mesh (same colors as interactive viewer)",
        max_faces_per_mesh=report_max_faces_per_mesh,
        include_placeholders=False,
    )
    placeholder_mesh_views = render_contract_part_assembly_views(
        contract,
        run_dir,
        assets_dir,
        "assembly_complete_view",
        title="Complete assembly with translucent bbox placeholders",
        max_faces_per_mesh=report_max_faces_per_mesh,
        include_placeholders=True,
        placeholder_alpha=0.18,
    )
    raw_render = raw_mesh_views.get("iso")
    placeholder_render = placeholder_mesh_views.get("iso")
    raw_assembly_asset = copy_mesh_asset_to_report(
        raw_assembly_path,
        assets_dir / "raw_hunyuan_assembly.obj",
    )
    placeholder_assembly_asset = copy_mesh_asset_to_report(
        placeholder_assembly_path,
        assets_dir / "placeholder_filled_assembly.obj",
    )

    rows = []
    contract_parts_by_id = contract_part_by_id(contract)
    for index, part in enumerate(manifest["parts"]):
        part_id = part["part_id"]
        target_class = part.get("target_class", "object")
        contract_part = contract_parts_by_id.get(part_id, {})
        display_label = display_label_for_part(contract_part, target_class)
        semantic_role = contract_part.get("semantic_role", "")
        display_group = semantic_role if target_class == "open_semantic_part" and semantic_role else target_class
        inferred = _infer_part_output_paths(run_dir, part_id, part)
        raw_output_path = inferred["raw_output_path"]
        normalized_output_path = inferred["normalized_output_path"]
        placeholder_output_path = inferred["placeholder_output_path"]
        display_mesh_path = normalized_output_path or placeholder_output_path
        mesh_caption = "normalized mesh" if normalized_output_path else "placeholder mesh"
        mesh_render = None
        if display_mesh_path:
            mesh_render = render_obj_mesh_strict(
                display_mesh_path,
                parts_assets_dir / f"{part_id}_mesh.png",
                title=target_class,
                color=part_color(part, index=index),
                max_faces=report_max_faces_per_mesh,
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
        raw_glb = _export_report_part_glb(
            display_mesh_path,
            parts_assets_dir / f"{part_id}.glb",
            part,
            index,
        )
        if raw_glb is None:
            raw_glb = copy_asset_to_report(
                inferred["glb_path"],
                parts_assets_dir / f"{part_id}.glb",
            )
        prompt = find_part_prompt(part_id, contract)
        t2i_metadata = find_part_t2i_metadata(part_id, contract) or {}
        prompt_trace = t2i_metadata.get("prompt_trace") or {}
        status = "success" if raw_output_path else (
            "placeholder" if placeholder_output_path and target_class == "wall" else "failed"
        )
        rows.append({
            "part_id": part_id,
            "target_class": target_class,
            "display_label": display_label,
            "semantic_role": semantic_role,
            "display_group": display_group,
            "status": status,
            "prompt": prompt,
            "t2i_status": t2i_metadata.get("status", "not_recorded"),
            "t2i_provider": t2i_metadata.get("provider", contract_part.get("t2i_provider", "")),
            "prompt_policy_version": prompt_trace.get("prompt_policy_version") or contract_part.get("prompt_policy_version", ""),
            "part_description_core": prompt_trace.get("part_description_core") or contract_part.get("part_description_core", ""),
            "part_visual_subject": prompt_trace.get("part_visual_subject") or contract_part.get("part_visual_subject", ""),
            "part_description_source": prompt_trace.get("part_description_source") or contract_part.get("part_description_source", ""),
            "part_context_tail": prompt_trace.get("part_context_tail") or contract_part.get("part_context_tail", ""),
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

    class_distribution = Counter(row["display_group"] for row in rows)
    failed_rows = [row for row in rows if row["status"] == "failed"]
    report_failures = _failure_payload_for_report(failures, rows)
    summary = {
        "run_dir": str(run_dir),
        "layout_id": manifest.get("layout_id"),
        "run_config": manifest.get("v6_changes", {}),
        "num_parts": len(rows),
        "num_success": sum(row["status"] in ("success", "placeholder") for row in rows),
        "num_failures": len(failed_rows),
        "num_failure_records": len(report_failures.get("failures", [])),
        "class_distribution": dict(sorted(class_distribution.items())),
        "layout_overview": rel(layout_overview, report_dir),
        "layout_xz": rel(layout_xz, report_dir),
        "layout_yz": rel(layout_yz, report_dir),
        "layout_text_callouts": rel(layout_text_callouts, report_dir),
        "layout_s1_style_multiview": rel(layout_s1_style_multiview, report_dir),
        "layout_s1_style_label_coverage": (
            rel(layout_s1_style_label_coverage, report_dir)
            if layout_s1_style_label_coverage and layout_s1_style_label_coverage.exists()
            else None
        ),
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
        "assembly_complete_model": (
            rel(assembly_complete_model, report_dir) if assembly_complete_model else None
        ),
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

    quality_summary = find_quality_summary(run_dir)
    if quality_summary:
        summary["quality_agent"] = quality_summary
    s2_agent_trace, s2_trace_path = find_s2_agent_trace(run_dir)
    if s2_agent_trace:
        summary["s2_agent_trace"] = {
            **s2_agent_trace,
            "trace_path": rel(s2_trace_path, report_dir),
        }
    repair_agent_trace, repair_trace_path = find_s2_repair_agent_trace(run_dir)
    if repair_agent_trace:
        summary["s2_repair_agent_trace"] = {
            **repair_agent_trace,
            "trace_path": rel(repair_trace_path, report_dir),
        }
    focused_critics, focused_critics_path = find_focused_visual_critics(run_dir)
    if focused_critics:
        summary["focused_visual_critics"] = {
            **focused_critics,
            "trace_path": rel(focused_critics_path, report_dir),
        }

    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (report_dir / "parts.json").write_text(json.dumps(parts_json, indent=2, sort_keys=True) + "\n")
    (report_dir / "index.html").write_text(render_html(summary, parts_json, report_failures))
    (report_dir / "report-sw.js").write_text(render_service_worker())
    return report_dir


def render_service_worker():
    return """const CACHE_NAME = '__CACHE_NAME__';
const CACHEABLE_EXTENSIONS = ['.glb', '.obj', '.png', '.jpg', '.jpeg'];

function isCacheable(url) {
  return CACHEABLE_EXTENSIONS.some((suffix) => url.pathname.toLowerCase().endsWith(suffix));
}

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || !isCacheable(url)) {
    return;
  }
  event.respondWith(
    caches.open(CACHE_NAME).then((cache) =>
      cache.match(event.request).then((cached) => {
        if (cached) {
          return cached;
        }
        return fetch(event.request).then((response) => {
          if (response && response.ok) {
            cache.put(event.request, response.clone());
          }
          return response;
        });
      })
    )
  );
});
""".replace("__CACHE_NAME__", f"buildingblock-report-{REPORT_ASSET_VERSION}")


def render_quality_agent_section(summary, quality):
    def asset_url(path):
        if not path:
            return ""
        text = str(path)
        if text == "#":
            return text
        separator = "&" if "?" in text else "?"
        return f"{text}{separator}v={REPORT_ASSET_VERSION}"

    if not quality:
        return f"{QUALITY_SECTION_START}{QUALITY_SECTION_END}"
    metric_summary = quality.get("metric_summary", {}) or {}
    evidence = quality.get("evidence", {}) or {}
    downloads = []
    for label, key in [
        ("metrics_summary.json", "metrics_path"),
        ("quality_findings.json", "findings_path"),
        ("repair_plan.json", "repair_plan_path"),
        ("quality_agent_trace.json", "trace_path"),
    ]:
        path = quality.get(key)
        if path:
            downloads.append(f'<a href="{html.escape(asset_url(path))}">{html.escape(label)}</a>')

    issue_table = ""
    findings_path = quality.get("findings_path")
    if findings_path:
        try:
            findings_abs = Path(summary["run_dir"]) / "report" / findings_path
            findings = load_json(findings_abs)
            rows = []
            for issue in findings.get("issues", []):
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(issue.get('issue_id', '')))}</td>"
                    f"<td>{html.escape(str(issue.get('labels', [])))}</td>"
                    f"<td>{html.escape(', '.join(str(item) for item in issue.get('part_ids', [])))}</td>"
                    f"<td>{html.escape(str(issue.get('issue_type', '')))}</td>"
                    f"<td>{html.escape(str(issue.get('severity', '')))}</td>"
                    f"<td>{html.escape(str(issue.get('recommended_action', '')))}</td>"
                    f"<td>{html.escape(str(issue.get('diagnosis', '')))}</td>"
                    "</tr>"
                )
            issue_table = (
                "<details open><summary>Quality findings</summary>"
                "<table class='qa-table'><thead><tr><th>id</th><th>labels</th><th>part_ids</th><th>type</th><th>severity</th><th>action</th><th>diagnosis</th></tr></thead>"
                "<tbody>"
                + ("".join(rows) if rows else '<tr><td colspan="7">No issues.</td></tr>')
                + "</tbody></table></details>"
            )
        except Exception as exc:
            issue_table = f"<p class='hint'>Could not load findings: {html.escape(repr(exc))}</p>"

    repair_table = ""
    repair_path = quality.get("repair_plan_path")
    if repair_path:
        try:
            repair_abs = Path(summary["run_dir"]) / "report" / repair_path
            repair_plan = load_json(repair_abs)
            rows = []
            for action in repair_plan.get("actions", []):
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(action.get('issue_id', '')))}</td>"
                    f"<td>{html.escape(str(action.get('labels', [])))}</td>"
                    f"<td>{html.escape(', '.join(str(item) for item in action.get('part_ids', [])))}</td>"
                    f"<td>{html.escape(str(action.get('planned_action', '')))}</td>"
                    f"<td>{html.escape(str(action.get('repair_instruction', '')))}</td>"
                    "</tr>"
                )
            repair_table = (
                "<details open><summary>Dry-run repair plan</summary>"
                "<table class='qa-table'><thead><tr><th>issue</th><th>labels</th><th>part_ids</th><th>planned action</th><th>instruction</th></tr></thead>"
                "<tbody>"
                + ("".join(rows) if rows else '<tr><td colspan="5">No planned actions.</td></tr>')
                + "</tbody></table></details>"
            )
        except Exception as exc:
            repair_table = f"<p class='hint'>Could not load repair plan: {html.escape(repr(exc))}</p>"

    evidence_figures = []
    for key, caption in [
        ("layout_text_callouts", "V9 evidence · layout callouts"),
        ("layout_overview", "V9 evidence · layout top"),
    ]:
        path = evidence.get(key)
        if path:
            evidence_figures.append(f'<figure><img src="{html.escape(asset_url(path))}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    assembly_views = evidence.get("assembly_geometry_views") or {}
    if isinstance(assembly_views, dict):
        for name in ("iso", "front", "side"):
            path = assembly_views.get(name)
            if path:
                evidence_figures.append(f'<figure><img src="{html.escape(asset_url(path))}"><figcaption>V9 evidence · assembly {html.escape(name)}</figcaption></figure>')
    partboards = evidence.get("partboards") or []
    if isinstance(partboards, list):
        for index, path in enumerate(partboards[:2]):
            if path:
                evidence_figures.append(f'<figure><img src="{html.escape(asset_url(path))}"><figcaption>V9 partboard {index}</figcaption></figure>')

    return f"""{QUALITY_SECTION_START}
        <section class="panel wide qa-section">
          <h2>V9 Agent QA / Repair Loop</h2>
          <p class="hint">First-phase infrastructure: deterministic mesh/layout metrics, optional mock visual findings, and dry-run local repair planning. No GPU generation is invoked here.</p>
          <div class="qa-kpis">
            <div><b>Scene score</b><span>{html.escape(str(quality.get('scene_score', 'metrics-only')))}</span></div>
            <div><b>Issues</b><span>{html.escape(str(quality.get('issue_count', 0)))}</span></div>
            <div><b>Planned actions</b><span>{html.escape(str(quality.get('planned_action_count', 0)))}</span></div>
            <div><b>Mesh gaps</b><span>{html.escape(str(metric_summary.get('mesh_gap_count', 'n/a')))}</span></div>
            <div><b>Flagged parts</b><span>{html.escape(str(metric_summary.get('flagged_part_count', 'n/a')))}</span></div>
          </div>
          <p>{html.escape(str(quality.get('global_summary', '')))}</p>
          <div class="downloads">{''.join(downloads)}</div>
          {issue_table}
          {repair_table}
          <details open>
            <summary>Evidence pack preview</summary>
            <div class="view-grid qa-evidence">{''.join(evidence_figures) if evidence_figures else '<p>No evidence images recorded.</p>'}</div>
          </details>
        </section>
        {QUALITY_SECTION_END}"""


def render_s2_agent_trace_section(summary, trace):
    def asset_url(path):
        if not path:
            return ""
        text = str(path)
        separator = "&" if "?" in text else "?"
        return f"{text}{separator}v={REPORT_ASSET_VERSION}"

    if not trace:
        return f"{S2_TRACE_SECTION_START}{S2_TRACE_SECTION_END}"

    trace_path = Path(summary["run_dir"]) / "agent_trace" / "s2_agent_trace.json"
    trace_rel = os.path.relpath(trace_path, Path(summary["run_dir"]) / "report") if trace_path.exists() else ""
    rows = []
    for part in trace.get("parts", []):
        t2i = part.get("t2i", {}) or {}
        geometry = part.get("geometry", {}) or {}
        texture = part.get("texture", {}) or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(part.get('label', '')))}</td>"
            f"<td>{html.escape(str(part.get('part_description_core') or part.get('part_description') or ''))}</td>"
            f"<td>{html.escape(str(part.get('semantic_role', '')))}</td>"
            f"<td>{html.escape(str(t2i.get('status', '')))}</td>"
            f"<td>{html.escape(str(geometry.get('status', '')))}</td>"
            f"<td>{html.escape(str(texture.get('status', '')))}</td>"
            f"<td>{html.escape(str(geometry.get('final_failure_type') or ''))}</td>"
            "</tr>"
        )
    summary_payload = trace.get("summary", {}) or {}
    policy = trace.get("agent_policy", {}) or {}
    downloads = f'<a href="{html.escape(asset_url(trace_rel))}">s2_agent_trace.json</a>' if trace_rel else ""
    return f"""{S2_TRACE_SECTION_START}
        <section class="panel wide qa-section">
          <h2>ArchStudio-S2 Agent Trace</h2>
          <p class="hint">当前 agent 是可审计的单次流程 agent：逐 part 记录 T2I → Hunyuan-Omni geometry → texture 的状态、prompt core/context 拆分和失败原因。后续多模态评估/重试 agent 将基于这个 trace 扩展。</p>
          <div class="qa-kpis">
            <div><b>Parts</b><span>{html.escape(str(summary_payload.get('num_parts', 0)))}</span></div>
            <div><b>T2I ok</b><span>{html.escape(str(summary_payload.get('num_t2i_succeeded', 0)))}</span></div>
            <div><b>Geometry ok</b><span>{html.escape(str(summary_payload.get('num_geometry_succeeded', 0)))}</span></div>
            <div><b>Texture ok</b><span>{html.escape(str(summary_payload.get('num_texture_succeeded', 0)))}</span></div>
          </div>
          <p><b>Policy:</b> {html.escape(str(policy.get('version', '')))} · {html.escape(str(policy.get('name', '')))}</p>
          <div class="downloads">{downloads}</div>
          <details open>
            <summary>Per-part pipeline status</summary>
            <table class="qa-table"><thead><tr><th>#</th><th>core subject</th><th>role</th><th>T2I</th><th>geometry</th><th>texture</th><th>failure</th></tr></thead>
            <tbody>{''.join(rows) if rows else '<tr><td colspan="7">No trace parts.</td></tr>'}</tbody></table>
          </details>
        </section>
        {S2_TRACE_SECTION_END}"""


def render_focused_visual_critics_section(summary, focused):
    def asset_url(path):
        if not path:
            return ""
        text = str(path)
        separator = "&" if "?" in text else "?"
        return f"{text}{separator}v={REPORT_ASSET_VERSION}"

    def focused_asset_url(path):
        # Focused critic cards must show the exact VLM input images.  Do not
        # route through display-enhanced copies here; otherwise it looks like
        # an extra generated artifact instead of the real input.
        return asset_url(path)

    if not focused:
        return f"{FOCUSED_CRITIC_SECTION_START}{FOCUSED_CRITIC_SECTION_END}"

    report_root = Path(summary["run_dir"]) / "report"
    run_root = Path(summary["run_dir"]).resolve()
    parts_by_id = {}
    try:
        for item in load_json(report_root / "parts.json"):
            if isinstance(item, Mapping) and item.get("part_id"):
                parts_by_id[str(item["part_id"])] = item
    except Exception:
        parts_by_id = {}

    def report_rel(path):
        if not path:
            return ""
        try:
            candidate = Path(str(path))
            if not candidate.is_absolute():
                return str(path)
            candidate = candidate.resolve()
            try:
                candidate.relative_to(report_root.resolve())
                return os.path.relpath(str(candidate), report_root)
            except ValueError:
                pass
            if candidate.exists() and candidate.is_file():
                try:
                    inside_run = candidate.relative_to(run_root)
                except ValueError:
                    inside_run = Path("focused_external") / candidate.name
                mirror = report_root / "assets" / "s2_agent_loop" / inside_run
                mirror.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, mirror)
                return os.path.relpath(str(mirror), report_root)
            return os.path.relpath(str(candidate), report_root)
        except Exception:
            return str(path)

    def load_answer(path):
        if not path:
            return {}
        try:
            payload = load_json(Path(str(path)))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    cards_by_step = {}
    for result in focused.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        step_no, step_title, step_desc = _focused_scope_step(result.get("scope"))
        answer_payload = load_answer(result.get("answer_path"))
        answer = answer_payload.get("answer", {}) if isinstance(answer_payload.get("answer"), dict) else {}
        issues = answer.get("issues", []) if isinstance(answer.get("issues"), list) else []
        issue_rows = []
        for issue in issues[:8]:
            if not isinstance(issue, dict):
                continue
            issue_rows.append(
                "<tr>"
                f"<td>{html.escape(str(issue.get('issue_type', '')))}</td>"
                f"<td>{html.escape(str(issue.get('severity', '')))}</td>"
                f"<td>{html.escape(str(issue.get('evidence', ''))[:240])}</td>"
                f"<td>{html.escape(str(issue.get('repair_hint', ''))[:360])}</td>"
                "</tr>"
            )
        figures = []
        for image in result.get("input_images", []) or []:
            path = image.get("copied_path") or image.get("source_path")
            if not path:
                continue
            rel_image_path = report_rel(path)
            url = html.escape(focused_asset_url(rel_image_path))
            figures.append(
                "<figure class='focused-figure'>"
                f"<a href='{url}' target='_blank' rel='noopener'><img src='{url}'></a>"
                f"<figcaption>{html.escape(str(image.get('label') or image.get('index')))}</figcaption>"
                "</figure>"
            )
        request_payload = load_answer(result.get("request_path"))
        prompt = request_payload.get("prompt", {}) if isinstance(request_payload.get("prompt"), dict) else {}
        question = prompt.get("question") or request_payload.get("question") or ""
        do_not_judge = prompt.get("do_not_judge") or []
        error_text = str(result.get("error") or answer_payload.get("error") or "")
        error_html = f"<p class='critic-error'><b>VLM call error:</b> {html.escape(error_text[:1600])}</p>" if error_text else ""
        card = (
            f"""
            <article class="focused-card step-card">
              <div class="critic-head">
                <div>
                  <div class="label">{html.escape(str(result.get('scope', '')))} · {html.escape(str(result.get('case_id', '')))}</div>
                  <h3>{html.escape(str(result.get('part_id') or 'scene assembly'))}</h3>
                </div>
                <div class="critic-score">
                  <span>score {html.escape(str(answer.get('score', result.get('score', ''))))}</span>
                  <span>{html.escape(str(answer.get('verdict', result.get('verdict', ''))))}</span>
                  <span>{html.escape(str(result.get('status', '')))}</span>
                </div>
              </div>
              <div class="step-io-grid">
                <div>
                  <h4>Input shown to VLM</h4>
                  <div class="focused-images">{''.join(figures) if figures else '<p>No focused critic images.</p>'}</div>
                </div>
                <div class="critic-copy">
                  <h4>Question</h4>
                  <p>{html.escape(str(question))}</p>
                  <h4>Do not judge in this step</h4>
                  <p>{html.escape(', '.join(str(item) for item in do_not_judge))}</p>
                  <h4>Answer</h4>
                  <p>{html.escape(str(answer.get('summary') or result.get('summary') or ''))}</p>
                  {error_html}
                </div>
              </div>
              <details open>
                <summary>Focused issues</summary>
                <table class="qa-table"><thead><tr><th>type</th><th>severity</th><th>evidence</th><th>repair hint</th></tr></thead>
                <tbody>{''.join(issue_rows) if issue_rows else '<tr><td colspan="4">No issues returned.</td></tr>'}</tbody></table>
              </details>
              <div class="downloads">
                <a href="{html.escape(asset_url(report_rel(result.get('request_path'))))}">request.json</a>
                <a href="{html.escape(asset_url(report_rel(result.get('answer_path'))))}">answer.json</a>
              </div>
            </article>
            """
        )
        cards_by_step.setdefault((step_no, step_title, step_desc), []).append(card)
    step_sections = []
    for (step_no, step_title, step_desc), cards in sorted(cards_by_step.items(), key=lambda item: item[0][0]):
        step_sections.append(
            f"""
            <section class="agent-step">
              <div class="step-heading">
                <span>{html.escape(step_no)}</span>
                <div><h3>{html.escape(step_title)}</h3><p>{html.escape(step_desc)}</p></div>
              </div>
              {''.join(cards)}
            </section>
            """
        )
    return f"""{FOCUSED_CRITIC_SECTION_START}
        <section class="panel wide qa-section focused-critics">
          <h2>Focused Visual Critics</h2>
          <p class="hint">修正版 agent 判别器：每个 step 只问一个 scope 的一个问题，不再把 layout/mesh/part/reference 全部混在一次 VLM 里。每张图都可点击放大；过暗的 renderer 图会生成 display-enhanced 展示副本，原始输入仍保留在 request/answer JSON 和 original 链接中。</p>
          <div class="qa-kpis">
            <div><b>Provider</b><span>{html.escape(str(focused.get('provider', '')))}</span></div>
            <div><b>Model</b><span>{html.escape(str(focused.get('model', '')))}</span></div>
            <div><b>Cases</b><span>{html.escape(str(len(focused.get('results', []) or [])))}</span></div>
            <div><b>Scopes</b><span>{html.escape(', '.join(str(item) for item in focused.get('scopes', []) or []))}</span></div>
          </div>
          {''.join(step_sections) if step_sections else '<p>No focused visual critic cases recorded.</p>'}
        </section>
        {FOCUSED_CRITIC_SECTION_END}"""


def render_s2_repair_agent_loop_section(summary, trace):
    def asset_url(path):
        if not path:
            return ""
        text = str(path)
        separator = "&" if "?" in text else "?"
        return f"{text}{separator}v={REPORT_ASSET_VERSION}"

    report_root = Path(summary["run_dir"]) / "report"

    def report_rel(path):
        if not path:
            return ""
        run_root = Path(summary["run_dir"]).resolve()
        try:
            candidate = Path(str(path))
            if not candidate.is_absolute():
                return str(path)
            candidate = candidate.resolve()
            try:
                candidate.relative_to(report_root.resolve())
                return os.path.relpath(str(candidate), report_root)
            except ValueError:
                pass
            if candidate.exists() and candidate.is_file():
                try:
                    inside_run = candidate.relative_to(run_root)
                except ValueError:
                    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in candidate.name)
                    inside_run = Path("external") / (safe_name or "artifact")
                mirror = report_root / "assets" / "s2_agent_loop" / inside_run
                mirror.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(candidate, mirror)
                    return os.path.relpath(str(mirror), report_root)
                except Exception:
                    return os.path.relpath(str(candidate), report_root)
            return os.path.relpath(str(candidate), report_root)
        except Exception:
            return str(path)

    def load_json_optional(path):
        if not path:
            return None
        try:
            candidate = Path(str(path))
            if candidate.exists():
                return load_json(candidate)
        except Exception:
            return None
        return None

    def text_optional(path, limit=9000):
        if not path:
            return ""
        try:
            candidate = Path(str(path))
            if candidate.exists():
                return candidate.read_text(encoding="utf-8", errors="ignore")[:limit]
        except Exception:
            return ""
        return ""

    def compact_prompt_questions(question_text, questions):
        if not questions:
            try:
                payload = json.loads(question_text)
                questions = payload.get("questions") or payload.get("instructions") or []
            except Exception:
                questions = []
        rows = []
        for index, question in enumerate(list(questions or [])[:8], start=1):
            rows.append(f"<li>{html.escape(str(question))}</li>")
        if rows:
            return "<ol class='critic-questions'>" + "".join(rows) + "</ol>"
        if question_text:
            return f"<pre class='critic-prompt'>{html.escape(question_text[:4000])}</pre>"
        return "<p>No evaluator question recorded.</p>"

    def _agent_image_caption(image):
        label = str(image.get("label") or image.get("index") or "image")
        lowered = label.lower()
        if "six_view_board" in lowered:
            return f"{label} · mesh 六视角合成图（iso/front/back/left/right/top），这是现在喂给 VLM 的主输入"
        if "layout_s1_style" in lowered or "s1_style" in lowered:
            return f"{label} · layout 使用 S1-style readable global-numeric 多视角板"
        if "reference" in lowered:
            return f"{label} · 单部件 T2I/reference 约束图"
        return label

    def render_agent_image_tiles(images, *, limit=18, css_class="agent-vlm-input-grid"):
        tiles = []
        for image in list(images or [])[:limit]:
            if not isinstance(image, Mapping):
                continue
            path = image.get("copied_path") or image.get("source_path")
            if not path:
                continue
            rel_image_path = report_rel(path)
            display_rel, enhanced, mean_luminance = _report_display_image_for_dark_render(
                report_root / rel_image_path if rel_image_path else path,
                report_root,
                rel_image_path,
            )
            image_url = html.escape(asset_url(display_rel))
            original_url = html.escape(asset_url(rel_image_path))
            caption = _agent_image_caption(image)
            caption_html = html.escape(caption)
            if enhanced:
                if mean_luminance is not None:
                    caption_html += html.escape(f" · display-enhanced mean={mean_luminance:.1f}")
                else:
                    caption_html += html.escape(" · display-enhanced")
                caption_html += f" · <a href='{original_url}' target='_blank' rel='noopener'>original</a>"
            tiles.append(
                "<figure class='agent-vlm-input-tile'>"
                f"<a href='{image_url}' target='_blank' rel='noopener'><img src='{image_url}'></a>"
                f"<figcaption>{caption_html}</figcaption>"
                "</figure>"
            )
        if not tiles:
            return "<p>No VLM input images recorded.</p>"
        return f"<div class='{html.escape(css_class)}'>" + "".join(tiles) + "</div>"

    def snapshot_path(snapshot, key):
        if not isinstance(snapshot, Mapping):
            return ""
        paths = snapshot.get("paths") if isinstance(snapshot.get("paths"), Mapping) else {}
        record = paths.get(key) if isinstance(paths.get(key), Mapping) else {}
        return str(record.get("path") or "") if record.get("exists") else ""

    def snapshot_prompt(snapshot):
        if not isinstance(snapshot, Mapping):
            return ""
        metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), Mapping) else {}
        return str(metadata.get("prompt_text") or "")

    def snapshot_record(snapshot, key):
        if not isinstance(snapshot, Mapping):
            return {}
        paths = snapshot.get("paths") if isinstance(snapshot.get("paths"), Mapping) else {}
        record = paths.get(key) if isinstance(paths.get(key), Mapping) else {}
        return record

    def snapshot_live_path(snapshot, key):
        record = snapshot_record(snapshot, key)
        path = record.get("live_path") or record.get("path")
        return str(path or "") if record.get("exists") and path else ""

    def file_signature(path):
        if not path:
            return ""
        try:
            candidate = Path(str(path))
            stat = candidate.stat()
            return f"{candidate.resolve()}::{stat.st_size}::{stat.st_mtime_ns}"
        except Exception:
            return str(path)

    def focused_case_artifact(iteration, part_id, scope, image_label_contains):
        evaluator = iteration.get("evaluator") if isinstance(iteration.get("evaluator"), Mapping) else {}
        focused = load_json_optional(evaluator.get("focused_summary_path")) or {}
        wanted_scope = str(scope or "")
        wanted_part = str(part_id or "")
        wanted_label = str(image_label_contains or "").lower()
        for result in focused.get("results") or []:
            if not isinstance(result, Mapping):
                continue
            if wanted_scope and str(result.get("scope") or "") != wanted_scope:
                continue
            if wanted_part and str(result.get("part_id") or "") != wanted_part:
                continue
            for image in result.get("input_images") or []:
                if not isinstance(image, Mapping):
                    continue
                label = str(image.get("label") or image.get("index") or "").lower()
                if wanted_label and wanted_label not in label:
                    continue
                path = image.get("copied_path") or image.get("source_path")
                if path:
                    return str(path)
        return ""

    def action_artifact_path(iteration, before, after, key, side, part_id=None, scope=None, image_label_contains=None):
        raw = snapshot_path(before if side == "before" else after, key)
        other = snapshot_path(after if side == "before" else before, key)
        if raw and other and file_signature(raw) != file_signature(other):
            return raw
        live_raw = snapshot_live_path(before if side == "before" else after, key)
        live_other = snapshot_live_path(after if side == "before" else before, key)
        if live_raw and live_other and file_signature(live_raw) != file_signature(live_other):
            return live_raw
        # Old traces captured mutable live paths only.  Fall back to the actual
        # focused VLM input image for the pre-action side so the report does not
        # misleadingly render the same final file twice.
        if side == "before" and part_id and scope and image_label_contains:
            focused_path = focused_case_artifact(iteration, part_id, scope, image_label_contains)
            if focused_path:
                return focused_path
        return raw or live_raw

    def before_after_same_note(before_path, after_path):
        if before_path and after_path and file_signature(before_path) == file_signature(after_path):
            return "<p class='critic-error'><b>对比证据警告：</b>before/after 仍指向同一个文件内容；此 run 可能是在修复快照逻辑前生成的旧 trace。</p>"
        return ""

    def render_artifact_image(path, caption):
        if not path:
            return (
                "<figure><div style='height:140px;display:flex;align-items:center;justify-content:center;"
                "background:#f1f3f6;border-radius:10px;color:#667'>missing</div>"
                f"<figcaption>{html.escape(caption)}</figcaption></figure>"
            )
        rel_path = report_rel(path)
        display_rel, enhanced, mean_luminance = _report_display_image_for_dark_render(
            report_root / rel_path if rel_path else path,
            report_root,
            rel_path,
        )
        note = ""
        if enhanced:
            note = f" · display-enhanced mean={mean_luminance:.1f}" if mean_luminance is not None else " · display-enhanced"
        return (
            "<figure>"
            f"<a href='{html.escape(asset_url(display_rel))}' target='_blank' rel='noopener'><img src='{html.escape(asset_url(display_rel))}'></a>"
            f"<figcaption>{html.escape(caption + note)}</figcaption>"
            "</figure>"
        )

    def render_mesh_board(path, part_id, stage_name, before_after):
        if not path:
            return ""
        mesh_path = Path(str(path))
        if not mesh_path.exists():
            return ""
        board_dir = report_root / "assets" / "s2_agent_loop" / "repair_mesh_boards" / _safe_asset_name(part_id)
        prefix = f"{_safe_asset_name(stage_name)}_{before_after}"
        try:
            views = render_obj_mesh_views(
                mesh_path,
                board_dir,
                prefix,
                title=f"{stage_name} {before_after} true mesh {part_id}",
                max_faces=22000,
                views=S2_VLM_SIX_VIEWS,
            )
            image_items = [(name, path) for name, path in views.items()]
            board = compose_image_grid(
                image_items,
                board_dir / f"{prefix}_six_view_board.png",
                f"{stage_name} {before_after} · true mesh six-view · {part_id}",
                columns=3,
            )
            return str(board) if board else str(views.get("iso") or "")
        except Exception:
            return ""

    def stage_part_record(stage, part_id):
        outputs = stage.get("outputs") if isinstance(stage.get("outputs"), Mapping) else {}
        results = outputs.get("results") if isinstance(outputs.get("results"), list) else []
        if not results:
            return {}
        return next((item for item in results if isinstance(item, Mapping) and str(item.get("part_id")) == str(part_id)), results[0])

    def render_generation_stage_card(iteration, stage, part_id):
        if not isinstance(stage, Mapping) or str(part_id) not in [str(item) for item in (stage.get("part_ids") or [])]:
            return ""
        outputs = stage.get("outputs") if isinstance(stage.get("outputs"), Mapping) else {}
        stage_name = str(stage.get("stage") or "")
        before = (outputs.get("before_artifacts") if isinstance(outputs.get("before_artifacts"), Mapping) else {}).get(str(part_id), {})
        after = (outputs.get("after_artifacts") if isinstance(outputs.get("after_artifacts"), Mapping) else {}).get(str(part_id), {})
        if stage_name == "prompt_repair":
            revisions = outputs.get("revisions") if isinstance(outputs.get("revisions"), list) else []
            rev = next((item for item in revisions if isinstance(item, Mapping) and str(item.get("part_id")) == str(part_id)), revisions[0] if revisions else {})
            if not isinstance(rev, Mapping):
                rev = {}
            new_prompt = str(rev.get("prompt_preview") or "")
            old_prompt = snapshot_prompt(before)
            repair_instruction = str(rev.get("repair_instruction") or "")
            links = []
            if rev.get("prompt_revision_path"):
                links.append(f"<a href='{html.escape(asset_url(report_rel(rev.get('prompt_revision_path'))))}'>agent_prompt_revision.json</a>")
            old_prompt_display = old_prompt or "not captured"
            new_prompt_display = new_prompt or "not captured"
            old_truncated = len(old_prompt_display) > 6000
            new_truncated = len(new_prompt_display) > 6000
            return (
                "<article class='repair-generation-card agent-action-card'>"
                f"<div class='label'>Agent action · prompt_repair · {html.escape(str(part_id))}</div>"
                "<h3>根据上一张 VLM revise 结果，改写下一次 T2I prompt</h3>"
                "<p class='hint'>这是一张独立的 Agent 动作卡：记录 Agent 实际准备写入/已写入的 prompt 变化，不是 VLM 输入的一部分。</p>"
                f"<p class='hint'>status={html.escape(str(rev.get('status', outputs.get('status', stage.get('status', '—')))))} · attempt={html.escape(str(rev.get('revision_attempt', '—')))}</p>"
                f"{('<p><b>来自 VLM 的修正指令：</b>' + html.escape(repair_instruction[:900]) + '</p>') if repair_instruction else ''}"
                "<div class='prompt-compare'>"
                f"<div><b>旧 prompt</b><pre>{html.escape(old_prompt_display[:6000])}{html.escape('\\n… truncated in report; open artifact for full text' if old_truncated else '')}</pre></div>"
                f"<div><b>新 prompt</b><pre>{html.escape(new_prompt_display[:6000])}{html.escape('\\n… truncated in report; open artifact for full text' if new_truncated else '')}</pre></div>"
                "</div>"
                f"<div class='downloads mini-downloads'>{''.join(links)}</div>"
                "</article>"
            )
        if stage_name == "t2i_generation":
            diff = (outputs.get("prompt_diffs") if isinstance(outputs.get("prompt_diffs"), Mapping) else {}).get(str(part_id), {})
            subprocesses = outputs.get("subprocesses") if isinstance(outputs.get("subprocesses"), list) else []
            proc = next((item for item in subprocesses if isinstance(item, Mapping) and str(part_id) in [str(x) for x in item.get("part_ids", [])]), {})
            command = proc.get("command") if isinstance(proc.get("command"), list) else outputs.get("command") if isinstance(outputs.get("command"), list) else []
            proc_status = str(proc.get("status") or stage.get("status") or outputs.get("status") or "unknown")
            after_caption = "新生成 T2I reference image" if proc_status == "succeeded" else "生成失败后的当前 reference（通常仍是旧图）"
            failure_note = ""
            if proc_status != "succeeded":
                failure_note = (
                    "<p class='critic-error'><b>T2I 没产生新图：</b>"
                    "subprocess 未成功结束，所以 before/after 可能指向同一个保留文件；请看 stderr_tail 和 t2i_metadata。</p>"
                )
            before_image = action_artifact_path(iteration, before, after, "reference_image", "before", part_id, "part_image", "part_reference")
            after_image = action_artifact_path(iteration, before, after, "reference_image", "after", part_id, "part_image", "part_reference")
            compare_note = before_after_same_note(before_image, after_image)
            return (
                "<article class='repair-generation-card agent-action-card'>"
                f"<div class='label'>Agent action · t2i_generation · {html.escape(str(part_id))}</div>"
                "<h3>执行 T2I 重新生成：对比旧 reference image 和新 reference image</h3>"
                "<p class='hint'>这是一张独立的生成结果卡：记录真实 subprocess、GPU、prompt 是否变化，以及生成前/后的图片证据。旧 run 若没有不可变快照，左图会回退到本轮 VLM 实际看到的旧 reference 输入。</p>"
                f"<p class='hint'>status={html.escape(proc_status)} · gpu={html.escape(str(proc.get('gpu', outputs.get('gpu', '—'))))} · prompt_changed={html.escape(str(diff.get('changed', 'unknown')))}</p>"
                f"{failure_note}{compare_note}"
                "<div class='before-after-grid'>"
                f"{render_artifact_image(before_image, '旧 T2I reference image')}"
                f"{render_artifact_image(after_image, after_caption)}"
                "</div>"
                f"<details><summary>T2I command</summary><pre class='mini-json'>{html.escape(' '.join(str(item) for item in command)[:3000])}</pre></details>"
                f"<details><summary>prompt diff</summary><pre class='mini-json'>{html.escape(json.dumps(diff, ensure_ascii=False, indent=2)[:4000])}</pre></details>"
                f"<details><summary>T2I subprocess result</summary><pre class='mini-json'>{html.escape(json.dumps(proc or outputs, ensure_ascii=False, indent=2)[:5000])}</pre></details>"
                "</article>"
            )
        if stage_name == "geometry_generation":
            result = stage_part_record(stage, part_id)
            before_mesh = (
                action_artifact_path(iteration, before, after, "normalized_mesh", "before", part_id, "part_mesh", "generated_true_part_mesh")
                or action_artifact_path(iteration, before, after, "raw_mesh", "before", part_id, "part_mesh", "generated_true_part_mesh")
            )
            after_mesh = action_artifact_path(iteration, before, after, "normalized_mesh", "after") or action_artifact_path(iteration, before, after, "raw_mesh", "after")
            before_board = focused_case_artifact(iteration, part_id, "part_mesh", "generated_true_part_mesh") if before_mesh and Path(str(before_mesh)).suffix.lower() not in {".obj"} else render_mesh_board(before_mesh, part_id, stage_name, "before")
            after_board = render_mesh_board(after_mesh, part_id, stage_name, "after")
            compare_note = before_after_same_note(before_mesh, after_mesh)
            command = []
            if isinstance(result, Mapping):
                attempts = result.get("attempts") if isinstance(result.get("attempts"), list) else []
                if attempts and isinstance(attempts[-1], Mapping) and isinstance(attempts[-1].get("command"), list):
                    command = attempts[-1].get("command")
            return (
                "<article class='repair-generation-card agent-action-card'>"
                f"<div class='label'>Agent action · geometry_generation · {html.escape(str(part_id))}</div>"
                "<h3>执行 I23D / mesh 重新生成：对比旧 mesh 和新 mesh 六视角</h3>"
                "<p class='hint'>这是一张独立的生成结果卡：记录真实 I23D 执行结果，并用 true mesh surface 六视角对比修改前后。</p>"
                f"<p class='hint'>status={html.escape(str(result.get('status', outputs.get('status', stage.get('status', '—'))) if isinstance(result, Mapping) else stage.get('status', '—')))} · gpu={html.escape(str(result.get('gpu', '—') if isinstance(result, Mapping) else '—'))}</p>"
                f"{compare_note}"
                "<div class='before-after-grid'>"
                f"{render_artifact_image(before_board, '旧 mesh 六视角 true surface board')}"
                f"{render_artifact_image(after_board, '新 mesh 六视角 true surface board')}"
                "</div>"
                f"<details><summary>I23D command/result</summary><pre class='mini-json'>{html.escape(json.dumps({'command': command, 'result': result}, ensure_ascii=False, indent=2)[:3500])}</pre></details>"
                "</article>"
            )
        return ""

    def render_generation_stage_cards_for_part(iteration, part_id, critic_scope=None):
        allowed_by_scope = {
            "part_image": {"prompt_repair", "t2i_generation"},
            "part_mesh": {"geometry_generation"},
        }
        allowed_stages = allowed_by_scope.get(str(critic_scope or ""), {"prompt_repair", "t2i_generation", "geometry_generation"})
        cards = [
            render_generation_stage_card(iteration, stage, part_id)
            for stage in iteration.get("stage_records", []) or []
            if isinstance(stage, Mapping) and str(stage.get("stage") or "") in allowed_stages
        ]
        return "".join(card for card in cards if card)

    def render_agent_stage_timeline(iteration):
        def stage_summary(stage):
            inputs = stage.get("inputs") if isinstance(stage.get("inputs"), Mapping) else {}
            outputs = stage.get("outputs") if isinstance(stage.get("outputs"), Mapping) else {}
            stage_name = str(stage.get("stage", ""))
            if stage_name == "assembly_validation":
                return (
                    f"VLM focused review collected {inputs.get('vlm_input_image_count', '—')} aggregate image references "
                    f"across scopes {', '.join(str(item) for item in inputs.get('focused_scopes', []) or []) or '—'}; "
                    f"phase gate={outputs.get('active_phase_gate', inputs.get('agent_phase', '—'))}; "
                    f"active findings={outputs.get('num_active_findings', '—')}/{outputs.get('num_findings', '—')}."
                )
            if stage_name == "prompt_repair":
                revisions = outputs.get("revisions") if isinstance(outputs.get("revisions"), list) else []
                statuses = Counter(str(item.get("status", "unknown")) for item in revisions if isinstance(item, Mapping))
                return f"Prompt repair plan for {len(revisions)} part(s); status {', '.join(f'{k}={v}' for k, v in sorted(statuses.items())) or outputs.get('status', '—')}."
            if stage_name == "t2i_generation":
                cmd = outputs.get("command") if isinstance(outputs.get("command"), list) else []
                part_ids = stage.get("part_ids") or []
                return f"T2I regeneration {'dry-run command' if outputs.get('status') == 'dry_run' else 'command'} for {len(part_ids)} part(s): {' '.join(str(item) for item in cmd[:3])}{' ...' if len(cmd) > 3 else ''}"
            if stage_name == "geometry_generation":
                results = outputs.get("results") if isinstance(outputs.get("results"), list) else []
                statuses = Counter(str(item.get("status", "unknown")) for item in results if isinstance(item, Mapping))
                return f"I23D/mesh regeneration for {len(results) or len(stage.get('part_ids') or [])} part(s); status {', '.join(f'{k}={v}' for k, v in sorted(statuses.items())) or outputs.get('status', '—')}."
            if outputs:
                status = outputs.get("status") or stage.get("status") or "recorded"
                return f"{stage_name or 'stage'} {status}; details available in raw JSON below."
            return f"{stage_name or 'stage'} {stage.get('status', '')}"

        rows = []
        for index, stage in enumerate(iteration.get("stage_records", []) or [], start=1):
            if not isinstance(stage, Mapping):
                continue
            inputs = stage.get("inputs") if isinstance(stage.get("inputs"), Mapping) else {}
            outputs = stage.get("outputs") if isinstance(stage.get("outputs"), Mapping) else {}
            primary = stage.get("primary_artifact")
            primary_link = f'<a href="{html.escape(asset_url(report_rel(primary)))}">artifact</a>' if primary else ""
            part_ids = stage.get("part_ids") or []
            part_preview = ", ".join(str(item) for item in part_ids[:3])
            if len(part_ids) > 3:
                part_preview += f" … +{len(part_ids) - 3}"
            raw_details = (
                "<details><summary>raw stage inputs/outputs JSON</summary>"
                f"<pre class='mini-json'>{html.escape(json.dumps({'inputs': inputs, 'outputs': outputs}, ensure_ascii=False, indent=2)[:4000])}</pre>"
                "</details>"
            )
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{html.escape(str(stage.get('stage', '')))}</td>"
                f"<td>{html.escape(str(stage.get('status', '')))}</td>"
                f"<td>{html.escape(part_preview)}</td>"
                f"<td>{html.escape(stage_summary(stage))}{primary_link}{raw_details}</td>"
                "</tr>"
            )
        if not rows:
            return "<p>No stage records for this iteration.</p>"
        return (
            "<table class='qa-table agent-stage-table'><thead><tr>"
            "<th>#</th><th>agent step</th><th>status</th><th>part ids</th><th>human-readable result</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )

    def render_iteration_io_summary(iteration):
        evaluator = iteration.get("evaluator") or {}
        focused = evaluator.get("focused_summary_path")
        answer = evaluator.get("answer_path")
        links = [
            f'<a href="{html.escape(asset_url(report_rel(focused)))}">focused case summary JSON</a>' if focused else "",
            f'<a href="{html.escape(asset_url(report_rel(answer)))}">converted findings JSON</a>' if answer else "",
        ]
        links = [item for item in links if item]
        focused_payload = load_json_optional(focused) or {}
        results = [result for result in (focused_payload.get("results") or []) if isinstance(result, Mapping)]
        scope_counts = Counter(str(result.get("scope", "unknown")) for result in results)
        status_counts = Counter(str(result.get("status", "unknown")) for result in results)
        image_counts = [len(result.get("input_images") or []) for result in results]
        per_call_image_text = ""
        if image_counts:
            per_call_image_text = f"min/avg/max {min(image_counts)}/{sum(image_counts) / len(image_counts):.1f}/{max(image_counts)} images per VLM call"
        legacy_input_preview = ""
        if not results and evaluator.get("input_images"):
            legacy_input_preview = (
                "<details open><summary>Legacy evaluator image references (fallback; not a focused per-call trace)</summary>"
                + render_agent_image_tiles(evaluator.get("input_images"), limit=3, css_class="agent-vlm-input-grid legacy-input-preview")
                + "</details>"
            )
        active_findings = iteration.get("active_findings") or []
        transition = iteration.get("workflow_transition") or {}
        transition_text = ""
        if isinstance(transition, Mapping) and transition:
            transition_text = f"{transition.get('from_phase', '')} → {transition.get('to_phase', '')}: {transition.get('reason', '')}; blocking={transition.get('blocking_finding_count', '')}"
        first_findings = []
        for finding in active_findings[:8]:
            if not isinstance(finding, Mapping):
                continue
            diagnosis = str(finding.get('diagnosis') or finding.get('evidence') or finding.get('summary') or '')
            repair_text = str(finding.get('repair_hint') or finding.get('recommended_action') or finding.get('repair_instruction') or finding.get('action') or '')
            first_findings.append(
                "<li>"
                f"<b>{html.escape(str(finding.get('issue_type', 'issue')))}</b> "
                f"sev={html.escape(str(finding.get('severity', '')))} · "
                f"part={html.escape(', '.join(str(item) for item in (finding.get('part_ids') or finding.get('labels') or []))[:180])} · "
                f"route={html.escape(str(finding.get('recommended_operation') or finding.get('repair_route') or ''))}<br>"
                f"<span class='review-opinion'>审核意见：{html.escape(diagnosis[:300])}</span>"
                f"{('<br><span class=\'repair-hint\'>修正建议：' + html.escape(repair_text[:300]) + '</span>') if repair_text else ''}"
                "</li>"
            )
        return f"""
          <article class="agent-iteration-io">
            <div class="critic-head">
              <div>
                <div class="label">Iteration {html.escape(str(iteration.get('iteration')))} · compact debug workflow status · scope {html.escape(str(evaluator.get('agent_scope', 'full_s2_repair')))} · phase {html.escape(str(evaluator.get('agent_phase', '')))}</div>
                <h3>本轮 Agent compact 状态（调试摘要，主线可跳过）</h3>
              </div>
              <div class="critic-score">
                <span>{html.escape(str(evaluator.get('mode', '')))}</span>
                <span>{html.escape(str(evaluator.get('status', '')))}</span>
                <span>{html.escape(str(iteration.get('num_active_findings', iteration.get('num_findings', 0))))} active</span>
              </div>
            </div>
            <p class="hint">这里不再堆叠所有图片，避免误解成“一次 VLM 调用了 80+ 张图”。真实输入在下面每个 Focused Visual Critic case 卡片内逐条展示：一个 case = 一次 VLM call = 对应 prompt + 2~3 张输入图 + 审核意见 + repair hint。</p>
            <div class="qa-kpis">
              <div><b>VLM calls</b><span>{html.escape(str(len(results)))}</span></div>
              <div><b>Scopes</b><span>{html.escape(', '.join(f'{k}={v}' for k, v in sorted(scope_counts.items())) or '—')}</span></div>
              <div><b>Statuses</b><span>{html.escape(', '.join(f'{k}={v}' for k, v in sorted(status_counts.items())) or '—')}</span></div>
              <div><b>Input/call</b><span>{html.escape(per_call_image_text or '—')}</span></div>
            </div>
            {f'<p class="hint"><b>Workflow transition:</b> {html.escape(transition_text)}</p>' if transition_text else ''}
            <div class="downloads">{''.join(links)}</div>
            <details><summary>本轮 active 审核意见 / repair routing 预览</summary><ul>{''.join(first_findings) if first_findings else '<li>No active findings for current phase.</li>'}</ul></details>
            {legacy_input_preview}
            <details><summary>Repair execution timeline: inputs → outputs</summary>{render_agent_stage_timeline(iteration)}</details>
          </article>
        """

    def render_9999_refresh_status():
        hook_path = Path(summary["run_dir"]) / "report" / "s2_9999_refresh.json"
        hook = load_json_optional(hook_path) or {}
        if not hook:
            return ""
        return (
            "<div class='agent-refresh-status'>"
            "<b>9999 refresh hook</b> "
            f"trigger={html.escape(str(hook.get('trigger', '')))} · "
            f"refreshed_at={html.escape(str(hook.get('refreshed_at', '')))} · "
            f"<a href='{html.escape(asset_url(report_rel(hook_path)))}'>s2_9999_refresh.json</a>"
            "</div>"
        )

    def issue_rows_from_answer(answer_payload, qa_payload):
        issues = []
        if isinstance(answer_payload, Mapping):
            findings = answer_payload.get("findings")
            if isinstance(findings, Mapping) and isinstance(findings.get("issues"), list):
                issues = findings.get("issues", [])
            elif isinstance(findings, list):
                issues = findings
        if not issues and isinstance(qa_payload, Mapping):
            qa_answer = qa_payload.get("answer")
            if isinstance(qa_answer, Mapping) and isinstance(qa_answer.get("findings"), list):
                issues = qa_answer.get("findings", [])
        rows = []
        for issue in issues[:20]:
            if not isinstance(issue, Mapping):
                continue
            labels = issue.get("labels") or issue.get("part_numbers") or []
            part_ids = issue.get("part_ids") or []
            diagnosis = issue.get("diagnosis") or issue.get("answer") or issue.get("rationale") or ""
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(issue.get('issue_id') or issue.get('issue_type') or 'issue'))}</td>"
                f"<td>{html.escape(str(issue.get('severity', '')))}</td>"
                f"<td>{html.escape(', '.join(str(item) for item in labels))}</td>"
                f"<td>{html.escape(str(issue.get('issue_type', '')))}</td>"
                f"<td>{html.escape(str(diagnosis))}</td>"
                f"<td>{html.escape(str(issue.get('recommended_action') or issue.get('action') or ''))}</td>"
                f"<td>{html.escape(', '.join(str(item) for item in part_ids)[:240])}</td>"
                "</tr>"
            )
        return rows

    def repair_rows_from_plan(plan_payload, operations):
        repairs = []
        if isinstance(plan_payload, Mapping) and isinstance(plan_payload.get("recommended_repairs"), list):
            repairs = plan_payload.get("recommended_repairs", [])
        if not repairs:
            repairs = operations or []
        rows = []
        for repair in repairs[:20]:
            if not isinstance(repair, Mapping):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(repair.get('action') or repair.get('operation_type') or repair.get('planned_action') or ''))}</td>"
                f"<td>{html.escape(str(repair.get('status', 'proposed')))}</td>"
                f"<td>{html.escape(', '.join(str(item) for item in (repair.get('part_numbers') or repair.get('labels') or [])))}</td>"
                f"<td>{html.escape(', '.join(str(item) for item in (repair.get('part_ids') or []))[:260])}</td>"
                f"<td>{html.escape(str(repair.get('instruction') or repair.get('repair_instruction') or repair.get('rationale') or repair.get('diagnosis') or '')[:520])}</td>"
                "</tr>"
            )
        return rows


    def render_prompt_part_context(prompt):
        context = prompt.get("part_context") if isinstance(prompt.get("part_context"), Mapping) else {}
        if not context:
            legacy_bbox = prompt.get("bbox") if isinstance(prompt.get("bbox"), Mapping) else {}
            context = {
                "layout_label": prompt.get("label"),
                "part_id": prompt.get("part_id"),
                "part_description": prompt.get("part_description"),
                "bbox_size_xyz": legacy_bbox.get("size") or legacy_bbox.get("extent"),
                "text_description": prompt.get("part_prompt") or prompt.get("part_description"),
                "semantic_role": prompt.get("semantic_role"),
                "layout_relations": prompt.get("spatial_relations"),
            }
        rows = []
        labels = [
            ("layout #", context.get("layout_label")),
            ("label guarantee", "same global numeric label as layout_s1_style_multiview_context" if context.get("layout_label") not in (None, "", [], {}) else None),
            ("bbox size XYZ", context.get("bbox_size_xyz")),
            ("part description", context.get("part_description")),
            ("text", context.get("text_description")),
            ("semantic role", context.get("semantic_role")),
            ("layout relations", context.get("layout_relations")),
        ]
        for key, value in labels:
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (list, tuple)):
                value_text = ", ".join(str(item) for item in value)
            else:
                value_text = str(value)
            rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(value_text[:900])}</td></tr>")
        if not rows:
            return ""
        return "<table class='part-context-table'><tbody>" + "".join(rows) + "</tbody></table>"

    def _issue_key_from_text(value):
        text = str(value or "").lower()
        for token in ("wrong_semantics", "whole_building", "background_or_base_artifact", "bad_aspect", "scale_mismatch", "low_geometry_quality", "wrong_orientation", "artifact_base_or_panel", "gap_or_seam", "overlap", "floating", "penetration", "missing_part"):
            if token in text:
                return token
        for token in ("semantic", "whole", "background", "aspect", "scale", "quality", "orientation", "artifact", "gap", "overlap", "floating", "penetration", "missing"):
            if token in text:
                return token
        return text[:80]

    def render_part_repair_flow(iteration, part_id, answer):
        if not part_id:
            return ""
        issue_types = set()
        for issue in answer.get("issues", []) if isinstance(answer.get("issues"), list) else []:
            if isinstance(issue, Mapping):
                issue_types.add(_issue_key_from_text(issue.get("issue_type")))
        cards = []
        for stage in iteration.get("stage_records", []) or []:
            if not isinstance(stage, Mapping) or str(part_id) not in [str(item) for item in (stage.get("part_ids") or [])]:
                continue
            outputs = stage.get("outputs") if isinstance(stage.get("outputs"), Mapping) else {}
            inputs = stage.get("inputs") if isinstance(stage.get("inputs"), Mapping) else {}
            stage_name = str(stage.get("stage") or "")
            operation_type = str(stage.get("operation_type") or "")
            failure_stage = str(stage.get("failure_stage") or "")
            source_issue = _issue_key_from_text(json.dumps({"operation_type": operation_type, "failure_stage": failure_stage, "outputs": outputs}, ensure_ascii=False)[:2000])
            # Keep this flow focused on the current VLM answer when possible;
            # always include single-part stages if issue matching is not encoded
            # in older traces.
            if issue_types and source_issue and source_issue not in issue_types and stage_name not in {"prompt_repair", "t2i_generation", "geometry_generation", "t2i_i3d_rerun_routed", "geometry_rerun_routed"}:
                continue
            summary = ""
            artifact = stage.get("primary_artifact")
            links = []
            if artifact:
                links.append(f"<a href='{html.escape(asset_url(report_rel(artifact)))}'>primary artifact</a>")
            if stage_name == "prompt_repair":
                revisions = outputs.get("revisions") if isinstance(outputs.get("revisions"), list) else []
                rev = next((item for item in revisions if isinstance(item, Mapping) and str(item.get("part_id")) == str(part_id)), revisions[0] if revisions else {})
                rev_path = rev.get("prompt_revision_path") if isinstance(rev, Mapping) else None
                rev_payload = load_json_optional(rev_path) or {}
                prompt_text = str(
                    rev.get("prompt_preview")
                    or rev_payload.get("prompt")
                    or ""
                )
                negative_text = str(
                    rev.get("negative_prompt_preview")
                    or rev_payload.get("negative_prompt")
                    or ""
                )
                repair_instruction = str(
                    rev.get("repair_instruction")
                    or rev_payload.get("repair_instruction")
                    or ""
                )
                subject_preview = str(
                    rev.get("subject_preview")
                    or rev_payload.get("subject")
                    or ""
                )
                summary = (
                    f"写入/计划写入 agent_prompt_revision attempt {rev.get('revision_attempt', '—') if isinstance(rev, Mapping) else '—'}; "
                    f"status={rev.get('status', outputs.get('status', '—')) if isinstance(rev, Mapping) else outputs.get('status', '—')}."
                )
                if rev_path:
                    links.append(f"<a href='{html.escape(asset_url(report_rel(rev_path)))}'>agent_prompt_revision.json</a>")
                extra = ""
                if subject_preview:
                    extra += f"<p><b>part subject:</b> {html.escape(subject_preview[:500])}</p>"
                if repair_instruction:
                    extra += f"<p><b>来自 VLM 的修正指令：</b>{html.escape(repair_instruction[:700])}</p>"
                if not prompt_text:
                    local_parts_by_id = {}
                    try:
                        for item in load_json(report_root / "parts.json"):
                            if isinstance(item, Mapping) and item.get("part_id"):
                                local_parts_by_id[str(item["part_id"])] = item
                    except Exception:
                        local_parts_by_id = {}
                    part = local_parts_by_id.get(str(part_id), {})
                    bbox = part.get('bbox') if isinstance(part.get('bbox'), Mapping) else {}
                    bbox_size = bbox.get('size') or bbox.get('extent') or []
                    prompt_text = (
                        f"Generate one isolated architectural component: {part.get('part_description') or part.get('source_actor_label') or part_id}. "
                        f"Role: {part.get('semantic_role') or part.get('target_class') or 'part'}. "
                        f"Layout bbox size XYZ: {', '.join(str(item) for item in bbox_size) if bbox_size else 'unknown'}. "
                        f"Text description: {part.get('part_prompt') or part.get('target_prompt') or part.get('part_description') or part_id}. "
                        f"Repair instruction: {repair_instruction or 'none'}."
                    )
                if prompt_text:
                    extra += f"<details><summary>修改后的 T2I prompt</summary><pre class='mini-json'>{html.escape(prompt_text[:2500])}</pre></details>"
                if negative_text:
                    extra += f"<details><summary>修改后的 negative prompt</summary><pre class='mini-json'>{html.escape(negative_text[:1800])}</pre></details>"
            elif stage_name == "t2i_generation":
                command = outputs.get("command") if isinstance(outputs.get("command"), list) else []
                summary = f"重新生成 T2I reference {'(dry-run 未真正调用)' if outputs.get('status') == 'dry_run' else ''}; status={outputs.get('status', stage.get('status', '—'))}."
                extra = f"<details><summary>T2I command</summary><pre class='mini-json'>{html.escape(' '.join(str(item) for item in command)[:2500])}</pre></details>" if command else ""
            elif stage_name == "geometry_generation":
                results = outputs.get("results") if isinstance(outputs.get("results"), list) else []
                part_result = next((item for item in results if isinstance(item, Mapping) and str(item.get("part_id")) == str(part_id)), results[0] if results else {})
                summary = f"重新生成 I23D/mesh {'(dry-run 未真正调用)' if part_result.get('status') == 'dry_run' else ''}; status={part_result.get('status', outputs.get('status', stage.get('status', '—')))}." if isinstance(part_result, Mapping) else f"重新生成 I23D/mesh; status={outputs.get('status', stage.get('status', '—'))}."
                extra = f"<details><summary>I23D command template / result</summary><pre class='mini-json'>{html.escape(json.dumps(part_result or outputs, ensure_ascii=False, indent=2)[:2500])}</pre></details>"
            else:
                summary = f"{stage_name}: status={stage.get('status', '—')}"
                extra = f"<details><summary>stage JSON</summary><pre class='mini-json'>{html.escape(json.dumps({'inputs': inputs, 'outputs': outputs}, ensure_ascii=False, indent=2)[:2500])}</pre></details>"
            cards.append(
                "<li>"
                f"<b>{html.escape(stage_name)}</b> · {html.escape(summary)}"
                f"<div class='downloads mini-downloads'>{''.join(links)}</div>"
                f"{extra}"
                "</li>"
            )
        if not cards:
            return ""
        return "<details open class='part-repair-flow'><summary>⑤ revise 后实际执行/计划的生成过程</summary><ol>" + "".join(cards) + "</ol></details>"

    def render_focused_visual_critic_block(iteration, next_iteration):
        evaluator = iteration.get("evaluator") or {}
        focused = load_json_optional(evaluator.get("focused_summary_path")) or {}
        answer_payload = load_json_optional(evaluator.get("answer_path")) or {}
        parts_by_id = {}
        try:
            for item in load_json(report_root / "parts.json"):
                if isinstance(item, Mapping) and item.get("part_id"):
                    parts_by_id[str(item["part_id"])] = item
        except Exception:
            parts_by_id = {}
        part_id_by_label = {
            int(item.get("label")): str(item.get("part_id"))
            for item in focused.get("part_index", []) or []
            if isinstance(item, Mapping)
            and str(item.get("label", "")).lstrip("-").isdigit()
            and item.get("part_id")
        }

        def focused_result_part_id(result, prompt):
            direct = result.get("part_id") or prompt.get("part_id")
            if direct:
                return str(direct)
            part_context = prompt.get("part_context") if isinstance(prompt.get("part_context"), Mapping) else {}
            context_part_id = part_context.get("part_id") if isinstance(part_context, Mapping) else None
            if context_part_id:
                return str(context_part_id)
            label = result.get("label") or prompt.get("label") or (part_context.get("layout_label") if isinstance(part_context, Mapping) else None)
            try:
                label_int = int(label)
            except (TypeError, ValueError):
                return ""
            return part_id_by_label.get(label_int, "")

        case_cards_by_step = {}
        for result in (focused.get("results") or []):
            if not isinstance(result, Mapping):
                continue
            step_no, step_title, step_desc = _focused_scope_step(result.get("scope"))
            request_payload = load_json_optional(result.get("request_path")) or {}
            response_payload = load_json_optional(result.get("answer_path")) or {}
            prompt = request_payload.get("prompt") if isinstance(request_payload.get("prompt"), Mapping) else {}
            answer = response_payload.get("answer") if isinstance(response_payload.get("answer"), Mapping) else {}
            raw_text = ""
            provider_metadata = response_payload.get("provider_metadata")
            if isinstance(provider_metadata, Mapping):
                raw_text = str(provider_metadata.get("raw_text") or provider_metadata.get("raw_response_text") or "")
            error_text = str(response_payload.get("error") or result.get("error") or "")
            figures = []
            for image in result.get("input_images", []) or []:
                path = image.get("copied_path") or image.get("source_path")
                if not path:
                    continue
                rel_image_path = report_rel(path)
                image_url = html.escape(asset_url(rel_image_path))
                figures.append(
                    "<figure class='critic-tile'>"
                    f"<a href='{image_url}' target='_blank' rel='noopener'><img src='{image_url}'></a>"
                    f"<figcaption>{html.escape(_agent_image_caption(image))}</figcaption>"
                    "</figure>"
                )
            local_issues = []
            for issue in answer.get("issues", []) if isinstance(answer.get("issues"), list) else []:
                if not isinstance(issue, Mapping):
                    continue
                local_issues.append(
                    "<li>"
                    f"<b>{html.escape(str(issue.get('issue_type', 'issue')))}</b> "
                    f"sev={html.escape(str(issue.get('severity', '')))} · "
                    f"{html.escape(str(issue.get('evidence', ''))[:260])} · "
                    f"<i>{html.escape(str(issue.get('repair_hint', issue.get('repair_instruction', '')))[:320])}</i>"
                    "</li>"
                )
            req_link = f'<a href="{html.escape(asset_url(report_rel(result.get("request_path"))))}">request.json</a>' if result.get("request_path") else ""
            ans_link = f'<a href="{html.escape(asset_url(report_rel(result.get("answer_path"))))}">answer.json</a>' if result.get("answer_path") else ""
            question_text = str(prompt.get('question', ''))
            do_not_judge_text = ', '.join(str(item) for item in (prompt.get('do_not_judge') or []))
            answer_summary = str(answer.get('summary') or result.get('summary') or '')
            verdict = str(answer.get('verdict', result.get('verdict', '')) or 'unknown')
            score = str(answer.get('score', result.get('score', '')) or '—')
            part_context_html = render_prompt_part_context(prompt)
            result_part_id = focused_result_part_id(result, prompt)
            generation_cards_html = ""
            if str(verdict).lower() in {"revise", "reject", "failed", "fail"}:
                generation_cards_html = render_generation_stage_cards_for_part(iteration, result_part_id, result.get("scope"))
            verdict_class = 'verdict-accept' if verdict.lower() == 'accept' else 'verdict-revise' if verdict.lower() in {'revise', 'reject', 'failed', 'fail'} else 'verdict-unknown'
            card = (
                "<article class='critic-qa-card focused-case'>"
                "<div class='critic-head'>"
                f"<div><div class='label'>单次 VLM call · {html.escape(str(result.get('scope', '')))} · {html.escape(str(result.get('case_id', '')))}</div>"
                f"<h3>{html.escape(str(result_part_id or result.get('part_id') or 'scene assembly'))}</h3></div>"
                f"<div class='critic-score'><span>input images {html.escape(str(len(result.get('input_images') or [])))}</span>"
                f"<span>score {html.escape(score)}</span>"
                f"<span class='{verdict_class}'>verdict: {html.escape(verdict)}</span>"
                f"<span>{html.escape(str(result.get('status', '')))}</span></div>"
                "</div>"
                "<div class='vlm-call-layout'>"
                "<section class='vlm-input-panel'>"
                "<h4>① 这一次 VLM 真实看到的图片</h4>"
                f"<p class='hint'>不是聚合图堆；本 call 只输入 {html.escape(str(len(result.get('input_images') or [])))} 张图。</p>"
                f"<div class='critic-images focused-images'>{''.join(figures) if figures else '<p>No input images recorded.</p>'}</div>"
                "</section>"
                "<section class='vlm-text-panel'>"
                "<h4>② VLM 输入文本核心信息</h4>"
                f"{part_context_html}"
                f"<p class='question-text'>{html.escape(question_text)}</p>"
                f"<p class='hint'><b>Do not judge:</b> {html.escape(do_not_judge_text or '—')}</p>"
                "<h4>③ VLM answer：审核结论 + 问题 + 修正建议</h4>"
                f"<div class='verdict-banner {verdict_class}'><b>{html.escape(verdict.upper())}</b><span>score {html.escape(score)}</span></div>"
                f"<p class='answer-summary'>{html.escape(answer_summary or 'No summary returned.')}</p>"
                f"{('<p class=\'critic-error\'><b>VLM call error:</b> ' + html.escape(error_text[:1200]) + '</p>') if error_text else ''}"
                f"<ul class='focused-issues'>{''.join(local_issues) if local_issues else '<li>No local focused issues.</li>'}</ul>"
                "</section>"
                "</div>"
                f"<details><summary>Debug: full prompt JSON sent to VLM</summary><pre class='critic-prompt'>{html.escape(json.dumps(prompt, ensure_ascii=False, indent=2)[:9000])}</pre></details>"
                f"<details><summary>Debug: raw GPT answer JSON</summary><pre class='critic-prompt'>{html.escape((raw_text or json.dumps(answer, ensure_ascii=False, indent=2))[:9000])}</pre></details>"
                f"<div class='downloads'>{req_link}{ans_link}</div>"
                "</article>"
            )
            case_cards_by_step.setdefault((step_no, step_title, step_desc), []).append(card)
            if generation_cards_html:
                case_cards_by_step.setdefault((step_no, step_title, step_desc), []).append(generation_cards_html)
        step_sections = []
        for (step_no, step_title, step_desc), cards in sorted(case_cards_by_step.items(), key=lambda item: item[0][0]):
            step_sections.append(
                f"""
                <section class="agent-step">
                  <div class="step-heading">
                    <span>{html.escape(step_no)}</span>
                    <div><h3>{html.escape(step_title)}</h3><p>{html.escape(step_desc)}</p></div>
                  </div>
                  {''.join(cards)}
                </section>
                """
            )
        all_results = focused.get("results") or []
        status_counts = Counter(str(result.get("status", "unknown")) for result in all_results if isinstance(result, Mapping))
        scope_counts = Counter(str(result.get("scope", "unknown")) for result in all_results if isinstance(result, Mapping))
        findings_rows = issue_rows_from_answer(answer_payload, {})
        current_score = iteration.get("scene_score")
        next_score = next_iteration.get("scene_score") if isinstance(next_iteration, Mapping) else ""
        summary_link = f'<a href="{html.escape(asset_url(report_rel(evaluator.get("focused_summary_path"))))}">focused_visual_critics_summary.json</a>' if evaluator.get("focused_summary_path") else ""
        answer_link = f'<a href="{html.escape(asset_url(report_rel(evaluator.get("answer_path"))))}">converted_answer.json</a>' if evaluator.get("answer_path") else ""
        return f"""
          <article class="critic-qa-card">
            <div class="critic-head">
              <div>
                <div class="label">Iteration {html.escape(str(iteration.get('iteration')))} · focused visual critics · scope {html.escape(str(evaluator.get('agent_scope', 'full_s2_repair')))} · {html.escape(str(evaluator.get('provider', '')))} / {html.escape(str(evaluator.get('model', '')))}</div>
                <h3>Focused Visual Critic QA</h3>
              </div>
              <div class="critic-score">
                <span>score {html.escape(str(current_score))}</span>
                <span>next {html.escape(str(next_score or '—'))}</span>
                <span>{html.escape(str(evaluator.get('status', '')))}</span>
              </div>
            </div>
            <p class="hint">Agent loop 内部的 focused evaluator：按真实执行 case 展示 VLM 输入/问题/回答/repair finding，便于追踪“判别器看到了什么、说了什么、下一步该改什么”。当 agent_scope=assemble_geometry 时，iteration 0 先逐个 part 审查 T2I/reference 与单体 mesh；后续 iteration 才进入整体拼装几何审查。Mesh 图为明亮 filled surface 真 mesh 预览，不使用点云/散点 fallback。</p>
            <p class="hint"><b>Layout label guarantee:</b> 每个 case 里的 <b>layout #</b> 就是右侧/输入图 <b>layout_s1_style_multiview_context</b> 中的同一个全局数字标号；二者都来自 contract parts 的顺序 index + 1。</p>
            <p class="hint"><b>本轮 focused case 总数：</b>{html.escape(str(len(all_results)))} · <b>scope:</b> {html.escape(', '.join(f'{k}={v}' for k, v in sorted(scope_counts.items())))} · <b>status:</b> {html.escape(', '.join(f'{k}={v}' for k, v in sorted(status_counts.items())))}</p>
            {''.join(step_sections) if step_sections else '<p>No focused critic cases recorded.</p>'}
            <details open>
              <summary>Converted reject/repair findings</summary>
              <table class="qa-table"><thead><tr><th>id</th><th>sev</th><th>#</th><th>type</th><th>diagnosis</th><th>action</th><th>part ids</th></tr></thead>
              <tbody>{''.join(findings_rows) if findings_rows else '<tr><td colspan="7">No converted findings.</td></tr>'}</tbody></table>
            </details>
            <div class="downloads">{summary_link}{answer_link}</div>
          </article>
        """

    def render_visual_critic_block(iteration, next_iteration):
        evaluator = iteration.get("evaluator") or {}
        if evaluator.get("mode") == "focused":
            return render_focused_visual_critic_block(iteration, next_iteration)
        qa_path = evaluator.get("visual_critic_qa_path")
        plan_path = evaluator.get("visual_critic_repair_plan_path")
        if not qa_path:
            interaction_dir = evaluator.get("interaction_dir")
            if interaction_dir:
                candidate = Path(str(interaction_dir)) / f"visual_critic_qa_iteration_{int(iteration.get('iteration', 0)):03d}.json"
                qa_path = str(candidate) if candidate.exists() else str(Path(str(interaction_dir)) / "visual_critic_qa.json")
        if not plan_path:
            interaction_dir = evaluator.get("interaction_dir")
            if interaction_dir:
                candidate = Path(str(interaction_dir)) / f"visual_critic_repair_plan_iteration_{int(iteration.get('iteration', 0)):03d}.json"
                plan_path = str(candidate) if candidate.exists() else str(Path(str(interaction_dir)) / "visual_critic_repair_plan.json")
        qa_payload = load_json_optional(qa_path) or {}
        plan_payload = load_json_optional(plan_path) or {}
        answer_payload = load_json_optional(evaluator.get("answer_path")) or {}
        question_text = text_optional(evaluator.get("question_prompt_path"))
        questions = evaluator.get("visual_critic_questions") or qa_payload.get("questions") or []
        board_path = evaluator.get("critic_board_path")
        if not board_path and isinstance(qa_payload, Mapping) and qa_payload.get("image"):
            board_candidate = Path(summary["run_dir"]) / str(qa_payload.get("image"))
            board_path = str(board_candidate) if board_candidate.exists() else qa_payload.get("image")
        image_links = []
        if board_path:
            board_url = html.escape(asset_url(report_rel(board_path)))
            image_links.append(
                "<figure class='critic-board'>"
                f"<a href='{board_url}' target='_blank' rel='noopener' title='Open full-size critic board'>"
                f"<img src='{board_url}'>"
                "</a>"
                "<figcaption>Single S2 critic board. Click image to open full size; browser zoom can then inspect labels/details.</figcaption>"
                "</figure>"
            )
        if not image_links:
            for image in (evaluator.get("input_images") or [])[:8]:
                path = image.get("copied_path") or image.get("source_path")
                if not path:
                    continue
                image_url = html.escape(asset_url(report_rel(path)))
                image_links.append(
                    "<figure class='critic-tile'>"
                    f"<a href='{image_url}' target='_blank' rel='noopener'><img src='{image_url}'></a>"
                    f"<figcaption>{html.escape(str(image.get('label') or image.get('index')))}</figcaption>"
                    "</figure>"
                )
        findings_rows = issue_rows_from_answer(answer_payload, qa_payload)
        repair_rows = repair_rows_from_plan(plan_payload, iteration.get("operations", []))
        findings = answer_payload.get("findings") if isinstance(answer_payload, Mapping) else {}
        if not isinstance(findings, Mapping):
            findings = qa_payload.get("answer", {}) if isinstance(qa_payload.get("answer"), Mapping) else {}
        current_score = iteration.get("scene_score")
        next_score = next_iteration.get("scene_score") if isinstance(next_iteration, Mapping) else ""
        summary_text = findings.get("global_summary") or findings.get("summary") or evaluator.get("status") or ""
        qa_link = f'<a href="{html.escape(asset_url(report_rel(qa_path)))}">visual_critic_qa.json</a>' if qa_path and Path(str(qa_path)).exists() else ""
        plan_link = f'<a href="{html.escape(asset_url(report_rel(plan_path)))}">visual_critic_repair_plan.json</a>' if plan_path and Path(str(plan_path)).exists() else ""
        answer_link = f'<a href="{html.escape(asset_url(report_rel(evaluator.get("answer_path"))))}">answer.json</a>' if evaluator.get("answer_path") else ""
        question_link = f'<a href="{html.escape(asset_url(report_rel(evaluator.get("question_prompt_path"))))}">question_prompt.txt</a>' if evaluator.get("question_prompt_path") else ""
        return f"""
          <article class="critic-qa-card">
            <div class="critic-head">
              <div>
                <div class="label">Iteration {html.escape(str(iteration.get('iteration')))} · {html.escape(str(evaluator.get('provider', '')))} / {html.escape(str(evaluator.get('model', '')))}</div>
                <h3>GPT-5.5-style Visual Critic QA</h3>
              </div>
              <div class="critic-score">
                <span>score {html.escape(str(current_score))}</span>
                <span>next {html.escape(str(next_score or '—'))}</span>
                <span>{html.escape(str(evaluator.get('status', '')))}</span>
              </div>
            </div>
            <div class="critic-grid">
              <div class="critic-images">{''.join(image_links) if image_links else '<p>No critic images recorded.</p>'}</div>
              <div class="critic-copy">
                <h4>Questions sent to the discriminator</h4>
                {compact_prompt_questions(question_text, questions)}
                <h4>Answer summary</h4>
                <p>{html.escape(str(summary_text or 'No textual summary recorded.'))}</p>
                <div class="downloads">{question_link}{answer_link}{qa_link}{plan_link}</div>
              </div>
            </div>
            <details open>
              <summary>Findings / reject reasons</summary>
              <table class="qa-table"><thead><tr><th>id</th><th>sev</th><th>#</th><th>type</th><th>diagnosis</th><th>action</th><th>part ids</th></tr></thead>
              <tbody>{''.join(findings_rows) if findings_rows else '<tr><td colspan="7">No visual findings recorded.</td></tr>'}</tbody></table>
            </details>
            <details open>
              <summary>Repair plan / executed operations</summary>
              <table class="qa-table"><thead><tr><th>action</th><th>status</th><th>#</th><th>part ids</th><th>instruction / rationale</th></tr></thead>
              <tbody>{''.join(repair_rows) if repair_rows else '<tr><td colspan="5">No repair actions recorded.</td></tr>'}</tbody></table>
            </details>
          </article>
        """

    if not trace:
        return ""

    score_rows = []
    for point in trace.get("score_trajectory", []) or []:
        score_rows.append(
            "<tr>"
            f"<td>{html.escape(str(point.get('iteration', '')))}</td>"
            f"<td>{html.escape(str(point.get('scene_score', '')))}</td>"
            f"<td>{html.escape(str(point.get('delta', '')))}</td>"
            f"<td>{html.escape(str(point.get('no_improvement_count', '')))}</td>"
            "</tr>"
        )
    taxonomy_rows = []
    for stage, count in sorted((trace.get("failure_taxonomy") or {}).items()):
        taxonomy_rows.append(
            f"<tr><td>{html.escape(str(stage))}</td><td>{html.escape(str(count))}</td></tr>"
        )
    rows = []
    interaction_blocks = []
    iteration_io_blocks = []
    for iteration in trace.get("iterations", []) or []:
        operations = iteration.get("operations", []) or []
        operation_text = "<br>".join(
            "{} · {} · {} · {}".format(
                html.escape(str(operation.get("operation_id", ""))),
                html.escape(str(operation.get("operation_type", ""))),
                html.escape(str(operation.get("status", ""))),
                html.escape(", ".join(str(item) for item in operation.get("part_ids", []) or [])),
            )
            for operation in operations
        ) or "none"
        stage_records = iteration.get("stage_records", []) or []
        stage_text = "<br>".join(
            "{} · {} · {}".format(
                html.escape(str(stage.get("stage", ""))),
                html.escape(str(stage.get("status", ""))),
                html.escape(str(stage.get("failure_stage") or "")),
            )
            for stage in stage_records[:8]
        )
        if len(stage_records) > 8:
            stage_text += f"<br>… +{len(stage_records) - 8} more"
        stage_text = stage_text or "none"
        phase = (iteration.get("evaluator") or {}).get("agent_phase", "")
        transition = iteration.get("workflow_transition") or {}
        transition_text = ""
        if isinstance(transition, Mapping) and transition:
            transition_text = f"{transition.get('from_phase', '')} → {transition.get('to_phase', '')}: {transition.get('reason', '')}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(iteration.get('iteration', '')))}</td>"
            f"<td>{html.escape(str(phase))}</td>"
            f"<td>{html.escape(str(iteration.get('accepted', '')))}</td>"
            f"<td>{html.escape(str(iteration.get('scene_score', '')))}</td>"
            f"<td>{html.escape(str(iteration.get('num_active_findings', iteration.get('num_findings', ''))))} / {html.escape(str(iteration.get('num_findings', '')))}</td>"
            f"<td>{operation_text}</td>"
            f"<td>{stage_text}<br>{html.escape(transition_text)}</td>"
            "</tr>"
        )
        evaluator = iteration.get("evaluator") or {}
        question = evaluator.get("question_prompt_path")
        answer = evaluator.get("answer_path")
        images = evaluator.get("input_images") or []
        image_links = []
        for image in images[:10]:
            path = image.get("copied_path") or image.get("source_path")
            if not path:
                continue
            image_url = html.escape(asset_url(report_rel(path)))
            image_links.append(
                "<figure style='margin:0'>"
                f"<a href='{image_url}' target='_blank' rel='noopener'>"
                f"<img src='{image_url}' style='width:160px;height:115px;object-fit:contain;background:#fff;border:1px solid #d7e2f5;border-radius:8px'>"
                "</a>"
                f"<figcaption style='font-size:11px;color:#555'>{html.escape(str(image.get('label') or image.get('index')))}</figcaption>"
                "</figure>"
            )
        iteration_io_blocks.append(render_iteration_io_summary(iteration))
        interaction_blocks.append(
            "<details>"
            f"<summary>Iteration {html.escape(str(iteration.get('iteration')))} evaluator I/O · "
            f"{html.escape(str(evaluator.get('status', '')))}</summary>"
            "<div class='downloads'>"
            f"{f'<a href=\"{html.escape(asset_url(report_rel(question)))}\">question prompt</a>' if question else ''}"
            f"{f'<a href=\"{html.escape(asset_url(report_rel(answer)))}\">answer JSON</a>' if answer else ''}"
            "</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:10px'>{''.join(image_links) if image_links else '<p>No evaluator images recorded.</p>'}</div>"
            "</details>"
        )
    critic_blocks = []
    iterations = trace.get("iterations", []) or []
    for index, iteration in enumerate(iterations):
        next_iteration = iterations[index + 1] if index + 1 < len(iterations) else None
        critic_blocks.append(render_visual_critic_block(iteration, next_iteration))
    trace_path = trace.get("trace_path")
    local_index = os.path.relpath(Path(summary["run_dir"]) / "agent_loop" / "index.html", Path(summary["run_dir"]) / "report")
    return f"""{S2_REPAIR_SECTION_START}
        <section class="panel wide qa-section">
          <h2>ArchStudio-S2 Reject/Accept Repair Agent</h2>
          <p class="hint">S1-style loop: validate → reject/accept → execute selected repair operations → rebuild assemblies/report → revalidate. It can snap mesh gaps, rerun selected T2I/geometry, and split box-like components into generic surface child parts.</p>
          <p class="hint"><b>Agent scope:</b> {html.escape(str((trace.get("policy") or {}).get("agent_scope", "full_s2_repair")))}. assemble_geometry 现在是状态机：先反复做 per-part T2I/reference 与单体 mesh 审查/修正/复查；只有所有 part-level active findings 清零后，才进入整体 scene_assembly 几何审查。</p>
          {render_9999_refresh_status()}
          <div class="qa-kpis">
            <div><b>Status</b><span>{html.escape(str(trace.get('status', '')))}</span></div>
            <div><b>Accepted</b><span>{html.escape(str(trace.get('accepted', False)))}</span></div>
            <div><b>Stop</b><span>{html.escape(str(trace.get('stop_condition', '')))}</span></div>
            <div><b>Score</b><span>{html.escape(str(trace.get('final_scene_score', '')))}</span></div>
          </div>
          <div class="downloads">
            {f'<a href="{html.escape(asset_url(trace_path))}">agent_loop/agent_trace.json</a>' if trace_path else ''}
            <a href="{html.escape(asset_url(local_index))}">agent_loop/index.html</a>
          </div>
          <details open>
            <summary>Reject/accept iterations / actual workflow state machine</summary>
            <p class="hint">findings 显示为 active/total：per-part 阶段只把 T2I/geometry findings 当作本轮 active repair；assembly findings 会等进入 global assembly 阶段再处理，因此不会在 part 没过时提前跳整体拼装。</p>
            <table class="qa-table"><thead><tr><th>iter</th><th>phase</th><th>accepted</th><th>score</th><th>findings</th><th>operations</th><th>stages / transition</th></tr></thead>
            <tbody>{''.join(rows) if rows else '<tr><td colspan="7">No repair-loop iterations recorded.</td></tr>'}</tbody></table>
          </details>
          <details>
            <summary>Score trajectory / auto-stop</summary>
            <table class="qa-table"><thead><tr><th>iter</th><th>score</th><th>delta</th><th>no-improvement</th></tr></thead>
            <tbody>{''.join(score_rows) if score_rows else '<tr><td colspan="4">No score trajectory.</td></tr>'}</tbody></table>
          </details>
          <details>
            <summary>Failure taxonomy</summary>
            <table class="qa-table"><thead><tr><th>stage</th><th>count</th></tr></thead>
            <tbody>{''.join(taxonomy_rows) if taxonomy_rows else '<tr><td colspan="2">No unresolved failure stages.</td></tr>'}</tbody></table>
          </details>
          <details>
            <summary>Agent VLM inputs / outputs / repair timeline (compact debug view; collapsed by default)</summary>
            {''.join(iteration_io_blocks) if iteration_io_blocks else '<p>No agent VLM I/O recorded.</p>'}
          </details>
          <details open>
            <summary>S1-style Visual Critic QA panels</summary>
            {''.join(critic_blocks) if critic_blocks else '<p>No visual critic QA recorded.</p>'}
          </details>
          <details>
            <summary>Raw evaluator I/O links</summary>
            {''.join(interaction_blocks) if interaction_blocks else '<p>No evaluator interactions recorded.</p>'}
          </details>
        </section>
        {S2_REPAIR_SECTION_END}"""


def render_html(summary, parts, failures):
    report_root = Path(summary["run_dir"]) / "report"

    def asset_url(path):
        if not path:
            return ""
        text = str(path)
        if text == "#":
            return text
        if Path(text.split("?", 1)[0]).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            text, _, _ = _report_display_image_for_dark_render(
                report_root / text,
                report_root,
                text,
                enabled=False,
            )
        separator = "&" if "?" in text else "?"
        return f"{text}{separator}v={REPORT_ASSET_VERSION}"

    def image_panel(title, image_path):
        return f"""
        <div class="panel">
          <h2>{html.escape(title)}</h2>
          <img src="{html.escape(asset_url(image_path))}">
        </div>
        """

    def model_or_image_panel(title, model_path=None, image_path=None, *, exposure="1.2", shadow_intensity="0.8"):
        escaped_title = html.escape(title)
        if model_path:
            return f"""
        <div class="panel">
          <h2>{escaped_title}</h2>
          <model-viewer src="{html.escape(asset_url(model_path))}" camera-controls auto-rotate interaction-prompt="none" exposure="{html.escape(exposure)}" shadow-intensity="{html.escape(shadow_intensity)}"></model-viewer>
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
                f"""<figure><img src="{html.escape(asset_url(views[name]))}"><figcaption>{html.escape(name)}</figcaption></figure>"""
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
              src="{html.escape(asset_url(part.get('model_glb')))}"
              camera-controls
              auto-rotate
              interaction-prompt="none"
              exposure="1.25"
              shadow-intensity="1"
              shadow-softness="0.6"
              environment-image="neutral"
              loading="lazy"
            ></model-viewer>
          </div>
        """ if part.get("model_glb") else f"""
          <div class="part-viewer-wrap">
            <div class="missing-model">No interactive mesh</div>
          </div>
        """
        cards.append(f"""
        <section class="card {html.escape(part['status'])}">
          <div class="meta">
            <h3>{html.escape(part.get('display_label') or part['target_class'])}</h3>
            <code>{html.escape(part['part_id'])}</code>
            <p class="status">{html.escape(part['status'])}</p>
          </div>
          <p class="hint"><b>role:</b> {html.escape(part.get('semantic_role') or 'n/a')} · <b>compat target:</b> {html.escape(part['target_class'])} · <b>T2I:</b> {html.escape(part.get('t2i_status') or 'not_recorded')} / {html.escape(part.get('t2i_provider') or 'n/a')}</p>
          <p class="hint"><b>visual subject:</b> {html.escape(part.get('part_visual_subject') or part.get('part_description_core') or 'n/a')} · <b>context removed:</b> {html.escape(part.get('part_context_tail') or 'n/a')}</p>
          <p class="prompt">{html.escape(part['prompt'])}</p>
          <div class="part-assets">
            <figure><img src="{html.escape(asset_url(part.get('reference_image')))}"><figcaption>reference image</figcaption></figure>
            <figure><img src="{html.escape(asset_url(part.get('layout_render')))}"><figcaption>layout highlight</figcaption></figure>
            {part_viewer}
          </div>
        </section>
        """)
    user_failures = [
        failure for failure in failures.get("failures", [])
        if failure.get("target_class") != "wall" and failure.get("failure_type") != "intentional_wall_placeholder"
    ]
    failures_html = "<p>No non-wall failures. Wall parts are intentionally represented by bbox boxes.</p>" if not user_failures else "<pre>{}</pre>".format(
        html.escape(json.dumps({**failures, "failures": user_failures}, indent=2))
    )
    class_dist = ", ".join(f"{k}: {v}" for k, v in summary["class_distribution"].items())
    run_config = summary.get("run_config") or {}
    run_config_items = [
        ("Scene", run_config.get("scene")),
        ("T2I provider", run_config.get("t2i_provider")),
        ("T2I model", run_config.get("t2i_model")),
        ("Wall policy", run_config.get("wall_policy")),
        ("Source layout", run_config.get("source_layout_abs")),
    ]
    run_config_html = "".join(
        f"<li><b>{html.escape(label)}:</b> {html.escape(str(value))}</li>"
        for label, value in run_config_items
        if value
    )
    if run_config_html:
        run_config_html = f"<ul>{run_config_html}</ul>"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ArchStudio-S2 Hunyuan Run Report</title>
  <script type="module">
    const localModelViewer = "{html.escape(asset_url(MODEL_VIEWER_LOCAL_ASSET))}";
    const cdnModelViewer = "{html.escape(MODEL_VIEWER_SOURCE)}";
    import(localModelViewer).catch(() => import(cdnModelViewer));
  </script>
  <script>
    if ('serviceWorker' in navigator) {{
      window.addEventListener('load', () => {{
        navigator.serviceWorker.register('report-sw.js').catch(() => {{}});
      }});
    }}
  </script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; background: #f7f7f5; color: #222; }}
    h1, h2 {{ margin-bottom: 0.3rem; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 18px 0; }}
    .panel, .card {{ background: white; border: 1px solid #ddd; border-radius: 12px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
    .panel img {{ width: 100%; height: 230px; object-fit: contain; background: #fff; }}
    .panel img.callout-img {{ height: auto; max-height: none; background: #f7f7f5; }}
    .panel model-viewer {{ height: 230px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(760px, 1fr)); gap: 16px; }}
    .card {{ border-left: 6px solid #999; }}
    .card.success {{ border-left-color: #54A24B; }}
    .card.placeholder {{ border-left-color: #4C78A8; }}
    .card.failed {{ border-left-color: #E45756; }}
    .meta {{ display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }}
    code {{ font-size: 11px; color: #555; }}
    .status {{ font-weight: 700; text-transform: uppercase; color: #555; }}
    .prompt {{ font-size: 13px; line-height: 1.45; background: #fafafa; padding: 10px 12px; border-radius: 8px; max-width: none; }}
    .part-assets {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; align-items: start; }}
    figure {{ margin: 0; }}
    figure img {{ width: 100%; height: 190px; object-fit: contain; background: #fff; border: 1px solid #eee; border-radius: 8px; }}
    figcaption {{ font-size: 12px; color: #666; text-align: center; }}
    .part-viewer-wrap {{
      width: 100%;
      min-width: 0;
      max-width: 100%;
      margin-top: 0;
    }}
    .part-viewer-wrap img,
    .part-viewer-wrap model-viewer,
    .part-viewer-wrap .missing-model {{
      width: 100%;
      height: 190px;
      margin-top: 0;
      object-fit: contain;
      border: 1px solid #eee;
      border-radius: 8px;
      background-color: #151922;
    }}
    .part-viewer-wrap .missing-model {{
      display: grid;
      place-items: center;
      color: #d9dde8;
      font-size: 13px;
      background:
        repeating-linear-gradient(45deg, #151922 0 12px, #1e2430 12px 24px);
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
    .hero .panel model-viewer {{ height: 420px; }}
    .wide {{ grid-column: 1 / -1; }}
    .view-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }}
    .view-grid img {{ width: 100%; height: 230px; object-fit: contain; background: #101318; border: 1px solid #222; border-radius: 8px; }}
    .downloads {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0 18px; }}
    .downloads a {{ background: #222; color: #fff; text-decoration: none; padding: 8px 12px; border-radius: 8px; font-size: 13px; }}
    .hint {{ color: #555; font-size: 13px; margin-top: 0; }}
    .qa-section {{ border-color: #b7c7e8; background: #f8fbff; }}
    .qa-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 12px 0; }}
    .qa-kpis div {{ background: white; border: 1px solid #d7e2f5; border-radius: 10px; padding: 10px; }}
    .qa-kpis b {{ display: block; color: #46628d; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    .qa-kpis span {{ font-size: 20px; font-weight: 800; }}
    .qa-table {{ width: 100%; border-collapse: collapse; margin: 10px 0 16px; font-size: 12px; }}
    .qa-table th, .qa-table td {{ border-bottom: 1px solid #d7e2f5; padding: 7px; text-align: left; vertical-align: top; }}
    .qa-table th {{ color: #46628d; text-transform: uppercase; letter-spacing: .04em; }}
    .qa-evidence img {{ background: #fff; }}
    .critic-qa-card {{ background: #fffdf7; border: 1px solid #d6c4a2; border-radius: 16px; padding: 16px; margin: 14px 0; box-shadow: 0 8px 22px rgba(41, 36, 23, .07); }}
    .critic-head {{ display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; margin-bottom: 12px; }}
    .critic-head h3 {{ margin: 4px 0 0; font-size: 22px; }}
    .critic-score {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }}
    .critic-score span {{ background: #223145; color: #fff; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}
    .critic-grid {{ display: grid; grid-template-columns: minmax(420px, 1.35fr) minmax(320px, .85fr); gap: 16px; align-items: start; }}
    .critic-images {{ display: grid; gap: 10px; }}
    .critic-board, .critic-tile {{ margin: 0; }}
    .critic-board img {{ width: 100%; max-height: 780px; object-fit: contain; background: #151922; border: 1px solid #222; border-radius: 12px; }}
    .critic-tile img {{ width: 100%; height: 180px; object-fit: contain; background: #151922; border: 1px solid #222; border-radius: 10px; }}
    .critic-board figcaption, .critic-tile figcaption {{ color: #665c4d; font-size: 12px; margin-top: 6px; }}
    .critic-copy {{ background: #f7f0df; border: 1px solid #ddcda9; border-radius: 12px; padding: 14px; }}
    .critic-copy h4 {{ margin: 0 0 8px; color: #4f3f24; }}
    .critic-questions {{ margin: 0 0 14px 20px; padding: 0; font-size: 13px; }}
    .critic-questions li {{ margin-bottom: 7px; }}
    .critic-prompt {{ max-height: 360px; overflow: auto; white-space: pre-wrap; background: #1e222b; color: #f4efe5; border-radius: 10px; padding: 12px; font-size: 11px; }}
    .focused-card {{ background: #fff; border: 1px solid #cfdbc8; border-radius: 16px; padding: 16px; margin: 14px 0; }}
    .focused-images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin: 12px 0; }}
    .focused-figure {{ margin: 0; }}
    .focused-figure img {{ width: 100%; height: 300px; object-fit: contain; background: #e8ecf2; border: 1px solid #9ca7b5; border-radius: 10px; }}
    .focused-figure figcaption {{ color: #4d5b48; font-size: 12px; margin-top: 6px; }}
    .agent-step {{ border: 1px solid #d6e2f2; background: #ffffff; border-radius: 18px; padding: 16px; margin: 18px 0; }}
    .step-heading {{ display: grid; grid-template-columns: auto 1fr; gap: 12px; align-items: start; margin-bottom: 12px; }}
    .step-heading > span {{ background: #17375f; color: #fff; border-radius: 999px; padding: 8px 12px; font-size: 12px; font-weight: 900; letter-spacing: .04em; }}
    .step-heading h3 {{ margin: 0 0 4px; font-size: 22px; color: #17375f; }}
    .step-heading p {{ margin: 0; color: #586a82; font-size: 13px; }}
    .step-io-grid {{ display: grid; grid-template-columns: minmax(420px, 1.2fr) minmax(320px, .8fr); gap: 16px; align-items: start; }}
    .step-io-grid h4 {{ margin: 0 0 8px; color: #385170; }}
    .focused-case .critic-tile img {{ height: 260px; background: #e8ecf2; border-color: #9ca7b5; }}
    .agent-refresh-status {{ background: #e9f7ef; border: 1px solid #9fd2b2; color: #214d32; border-radius: 10px; padding: 10px 12px; margin: 10px 0; font-size: 13px; }}
    .agent-iteration-io {{ background: #ffffff; border: 2px solid #98b7e4; border-radius: 18px; padding: 16px; margin: 16px 0; }}
    .agent-vlm-input-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; margin: 14px 0; }}
    .agent-vlm-input-tile {{ margin: 0; background: #f8fbff; border: 1px solid #c4d5ef; border-radius: 12px; padding: 10px; }}
    .agent-vlm-input-tile img {{ width: 100%; height: 340px; object-fit: contain; background: #edf2f8; border: 1px solid #a9b9cf; border-radius: 10px; }}
    .agent-vlm-input-tile figcaption {{ color: #34465f; font-size: 12px; margin-top: 7px; text-align: left; }}
    .mini-json {{ max-height: 180px; overflow: auto; white-space: pre-wrap; background: #f4f7fb; border: 1px solid #d7e2f5; border-radius: 8px; padding: 8px; font-size: 11px; color: #203047; }}
    .vlm-call-layout {{ display: grid; grid-template-columns: minmax(320px, 1.1fr) minmax(320px, .9fr); gap: 16px; align-items: start; margin: 12px 0; }}
    .vlm-input-panel, .vlm-text-panel {{ min-width: 0; background: #fff; border: 1px solid #e3d4b9; border-radius: 14px; padding: 12px; }}
    .vlm-input-panel h4, .vlm-text-panel h4 {{ margin: 0 0 8px; color: #385170; }}
    .question-text {{ background: #eef5ff; border-left: 4px solid #4078c0; border-radius: 8px; padding: 10px; font-weight: 650; }}
    .answer-summary {{ background: #eef9ef; border-left: 4px solid #3d8b50; border-radius: 8px; padding: 10px; }}
    .focused-issues {{ padding-left: 20px; }}
    .focused-issues li {{ margin: 8px 0; }}
    .review-opinion {{ color: #23364f; }}
    .repair-hint {{ color: #7a3d00; font-weight: 650; }}
    .part-context-table {{ width: 100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 12px; table-layout: fixed; }}
    .part-context-table th {{ width: 130px; color: #385170; text-align: left; vertical-align: top; padding: 6px; border-bottom: 1px solid #d7e2f5; }}
    .part-context-table td {{ padding: 6px; border-bottom: 1px solid #d7e2f5; overflow-wrap: anywhere; }}
    .verdict-banner {{ display:flex; justify-content:space-between; gap:10px; align-items:center; border-radius:12px; padding:10px 12px; margin:8px 0; }}
    .verdict-accept {{ background:#1f7a3a !important; color:#fff !important; }}
    .verdict-revise {{ background:#a84d00 !important; color:#fff !important; }}
    .verdict-unknown {{ background:#53606f !important; color:#fff !important; }}
    .part-repair-flow {{ margin-top: 12px; border: 1px solid #ecd6ae; background: #fffaf0; border-radius: 12px; padding: 10px; }}
    .part-repair-flow ol {{ padding-left: 22px; margin: 8px 0; }}
    .part-repair-flow li {{ margin: 10px 0; }}
    .mini-downloads {{ margin: 6px 0; }}
    .mini-downloads a {{ font-size: 11px; padding: 5px 8px; }}
    .agent-stage-table td {{ max-width: 420px; }}
    .label {{ color: #6e5a34; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; }}
    @media (max-width: 840px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .part-assets {{ grid-template-columns: 1fr; }}
      .part-viewer-wrap {{ width: 100%; min-width: 0; }}
      .critic-grid {{ grid-template-columns: 1fr; }}
      .vlm-call-layout {{ grid-template-columns: 1fr; }}
      .critic-head {{ flex-direction: column; }}
      .step-io-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>ArchStudio-S2 Hunyuan Run Report</h1>
  <p><b>Project:</b> ArchStudio: A Multi-Agent Framework for Layout-Guided 3D Architectural Asset Generation</p>
  <p><b>Stage:</b> Stage 2 / ArchStudio-S2 · Layout-guided 3D architectural asset generation</p>
  <p><b>Run:</b> {html.escape(summary['run_dir'])}</p>
  <p><b>Parts:</b> {summary['num_parts']} | <b>Success:</b> {summary['num_success']} | <b>Failures:</b> {summary['num_failures']} | <b>Failure records:</b> {summary.get('num_failure_records', summary['num_failures'])}</p>
  <p><b>Class distribution:</b> {html.escape(class_dist)}</p>
  <div class="panel">{run_config_html}</div>
  {render_s2_agent_trace_section(summary, summary.get('s2_agent_trace'))}
  {render_focused_visual_critics_section(summary, summary.get('focused_visual_critics'))}
  {render_s2_repair_agent_loop_section(summary, summary.get('s2_repair_agent_trace'))}
  {render_quality_agent_section(summary, summary.get('quality_agent'))}
  <div class="downloads">
    <a href="{html.escape(asset_url(summary.get('assembly_raw_obj') or '#'))}">Download raw assembled OBJ</a>
    <a href="{html.escape(asset_url(summary.get('assembly_placeholder_obj') or '#'))}">Download complete assembly OBJ</a>
  </div>
  <h2>Overall 3D layout + assembled mesh</h2>
  <div class="hero">
    {model_or_image_panel('Large 3D layout boxes', summary.get('layout_3d_model'), summary.get('layout_3d'), exposure='1.2', shadow_intensity='0.7')}
    {model_or_image_panel('Large assembled mesh (with shrunken wall)', summary.get('assembly_complete_model') or summary.get('assembly_raw_model'), summary.get('assembly_placeholder_image') or summary.get('assembly_raw_image'))}
  </div>
  <section class="panel wide">
    <h2>Layout S1-style numeric labels</h2>
    <p class="hint">Layout visualization now follows the S1 readable multi-view style, but uses plain global numeric labels only; prompt text remains in the part cards instead of callout labels.</p>
    <img class="callout-img" src="{html.escape(asset_url(summary.get('layout_s1_style_multiview') or summary.get('layout_text_callouts')))}">
  </section>
  <div class="summary">
    {view_gallery('Layout multi-view renders', summary.get('layout_3d_views', {}))}
    {view_gallery('Assembled mesh multi-view renders', summary.get('assembly_placeholder_views', {}) or summary.get('assembly_raw_views', {}))}
  </div>
  <div class="summary">
    <div class="panel"><h2>Layout XY top</h2><img src="{html.escape(asset_url(summary['layout_overview']))}"></div>
    <div class="panel"><h2>Layout XZ elevation</h2><img src="{html.escape(asset_url(summary['layout_xz']))}"></div>
    <div class="panel"><h2>Layout YZ elevation</h2><img src="{html.escape(asset_url(summary['layout_yz']))}"></div>
    {model_or_image_panel('Layout 3D bbox', summary.get('layout_3d_model'), summary.get('layout_3d'), exposure='1.2', shadow_intensity='0.7')}
    {model_or_image_panel('Raw assembly (no wall placeholder)', summary.get('assembly_raw_model'), summary.get('assembly_raw_image') or summary.get('assembly_placeholder_image'))}
    {model_or_image_panel('Complete assembly (wall bbox translucent)', summary.get('assembly_complete_model'), summary.get('assembly_placeholder_image'))}
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
