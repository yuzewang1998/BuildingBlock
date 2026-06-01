# Stage2 with Agentic — 暂停/恢复工作 README

> 暂定名：**Stage2 with Agentic**  
> 写作目的：记录当前 Stage2 引入 Agent 的目标、已实现代码、9999 可视化方式、实验现象、已知问题和下一步规划。项目短期暂停后，可以从本文快速恢复上下文。

## 1. 一句话目标

Stage2 with Agentic 不是重写原有 BuildingBlock Stage2 流程，而是在现有 **S1 part-level layout + description → T2I → I23D → Assemble → Texture** 流程外，加入一个可审查、可迭代、可回退、可视化的 Agent 层：

- 每一步都能看到 Agent 实际看了什么输入；
- VLM 的 request / answer / accept-or-revise 结论可以审计；
- 若不通过，Agent 能按原因路由到 T2I prompt 修正、I23D prompt/geometry 重生成、或 assemble 局部修复；
- 每次迭代产生的中间变量都要保留，方便前后对比和回退；
- 9999 页面要成为主要调试入口，而不是只展示最终结果。

## 2. Stage2 原始流程边界

当前 Stage2 消费 Stage1 给出的 **part-level layout with detailed description**。Stage1 的 layout 已经包含每个 part 的 bbox、类别/文本描述，以及全局布局语义。

Stage2 原始顺序保持不变：

1. **T2I / part reference image**  
   根据 S1 中每个 layout part 的 text description 生成 part-level image。
2. **I23D / I3D geometry**  
   根据 part bbox + part image + prompt，用 Hunyuan3D-Omni 生成 3D mesh。
3. **Assemble**  
   按 layout bbox 把所有 part mesh 归一化、摆放、拼装成整体建筑。
4. **Texture**  
   后续 Hunyuan3D-Paint / texture 生成。

Agent 的原则：**先不改变大流程，只在流程节点之间插入审核与修正闭环**。

## 3. Agentic 设计：分层审核与修正闭环

### 3.1 T2I 节点：part image 审查 / 修正

目标：对每个 part 逐个检查，不应先整体拼装。

每个 part 的 T2I 审查输入应保持聚焦：

- 当前 part 的 reference image；
- S1-style layout multiview context，且突出当前 part；
- 当前 part 在 layout 渲染中的数字标号；
- bbox size / aspect ratio，不强调 bbox center；
- part text description；
- 简洁 required JSON schema。

VLM 需要回答：

- 是否 accept；
- 如果 revise，问题是什么；
- 修正建议是什么；
- 问题更像 semantic prompt、scale/aspect、background/scene leakage，还是生成质量。

若 revise：

1. 根据 VLM 意见改写 T2I prompt；
2. 重新生成 reference image；
3. 保留 old/new image、old/new prompt、diff、T2I metadata、stderr_tail；
4. 用新图再次进入同一个 part 的 VLM 审查；
5. 直到 accept 或该阶段 max iteration 耗尽；
6. accept 后进入该 part 的 mesh 审查，阶段计数清零。

### 3.2 I23D 节点：part mesh 审查 / 修正

T2I accept 后，再检查对应 part 的 mesh。

输入：

- accepted reference image；
- 当前 mesh 的 true surface six-view render；
- S1-style layout multiview context，突出当前 part；
- bbox size / aspect ratio；
- part description；
- 可选 I23D prompt / generation metadata。

VLM 要判断：

- mesh 是否符合图、文、bbox shape family；
- 是否轴向错位、长宽深比例错误、塌缩、生成成柱/台阶/地面/门洞等错误结构；
- root cause 是 `source_image`、`i23d_generation`、`layout_contract` 还是 unclear。

修正路由：

- 若 source image 有问题，退回 T2I prompt repair；
- 若 I23D prompt 或 Hunyuan 生成问题，重写 I23D prompt 并重新生成 mesh；
- 若只是 assemble 归一化/轴向/pose 问题，进入 deterministic mesh alignment / assemble repair。

### 3.3 Assemble 节点：整体几何审查 / 修正

当单 part image 与 mesh 都通过后，才进入整体拼装检查。

输入：

- assembled building 的 colored true-surface six-view render；
- 每个 part 用不同颜色和数字 label 标明；
- part index / layout JSON context；
- S1-style layout board 可作为空间参考。

VLM 检查：

- 缝隙、漂浮、穿插、重叠；
- part 之间物理关系不合理；
- 局部 part 需要退回 I23D 或 T2I；
- 建议必须尽量落到具体 part id / label。

第一阶段 assemble agent 已经做过 mini-run 验证；当前更重要的是把前面的 per-part T2I / mesh loop 跑通。

## 4. 当前已实现的关键能力

### 4.1 核心代码入口

| 功能 | 文件 |
| --- | --- |
| Agent loop CLI | `scripts/run_archstudio_s2_agent_loop.py` |
| Focused visual critic / VLM request 构造 | `scene_synthesis/building_mesh/quality_agent.py` |
| Repair planning / T2I prompt repair / operation execution | `scene_synthesis/building_mesh/s2_repair_agent.py` |
| Agent trace schema | `scene_synthesis/building_mesh/s2_agent_trace.py` |
| 9999 report bundle / Agent visualization | `scene_synthesis/building_mesh/report_bundle.py` |
| S1-style layout multiview / mesh six-view render | `scene_synthesis/building_mesh/visualization.py` |
| local Qwen-Image T2I provider | `scene_synthesis/building_mesh/reference_images.py` |
| batch / helper scripts | `scripts/archstudio_s2_run_all_partfocus.py`, `scripts/run_archstudio_s2_focused_critics.py`, `scripts/evaluate_archstudio_s2_quality.py` |
| tests | `scripts/tests/test_archstudio_s2_quality_agent.py`, `scripts/tests/test_archstudio_s2_repair_agent.py` |

### 4.2 9999 可视化已经支持的内容

9999 报告页现在重点展示：

- **Focused Visual Critic QA**：按 Step 1 / Step 2 / Assemble 显示，而不是把所有图混在一起；
- **单次 VLM call**：展示 scope、part id、request JSON、answer JSON、accept/revise；
- **VLM input images**：展示真实送给 VLM 的图片；
- **Agent action · t2i_generation**：展示 T2I 重新生成动作、old/new prompt、old/new reference image、metadata；
- **Agent action · geometry_generation**：展示 mesh 重生成动作、old/new mesh render、metadata；
- **Iteration summary**：展示每轮 status、score、active findings、operations；
- **refresh hook**：Agent trace 更新后刷新 `report/index.html`，使 9999 能及时看到最新状态。

用户明确要求：以后每次更新/实验都要给出 9999 链接。

### 4.3 S1-style layout context

已经去掉了用户不想要的 `part_geometry_contract_bbox_proxy` 输入。现在 focused VLM 输入使用 **S1-style layout multiview context**，并突出当前 part。

注意：

- layout 标号应使用正常数字，不额外加 A/B/C/D/E；
- 当前 part 的 label 必须和右侧 `layout_s1_style_multiview_context` 中标号一致；
- 输入给 VLM 的 part 信息要摘核心，避免 raw JSON 干扰：bbox size、part description、layout label、required_json 最重要。

### 4.4 T2I loop 已经被修正为“同一 part 内部循环”

之前的问题：part image revise 后，重新生成了图，但没有用新图重新审查，而是跳到下一个 part。

现在逻辑应为：

1. part image revise；
2. repair T2I prompt；
3. rerun T2I；
4. 用新图重新 VLM 审查同一个 part；
5. accept 后才进入 mesh 阶段或下一个 part。

最近真实 run 中观察到 part #0 的 T2I 经过多次 revise 后 accept，说明这个闭环方向是对的。

### 4.5 Geometry-contract-aware T2I prompt repair

为了避免 case-by-case prompt patch，新增了系统性 prompt policy：

- 从 bbox size / aspect ratio 推断 shape family；
- 对 tall upright thin wall / gallery bar 等类型，prompt 强调几何 primitive、z-up、long horizontal extent、shallow depth；
- 随 retry attempt 逐步从语义丰富收敛到更适合 I23D 的 silhouette / primitive prompt；
- 明确禁止 stairs、plinth、paving、portal、columns、scene/background 等常见误生成。

当前 policy 名称：`geometry_contract_aware_i23d_prior_repair_v2`。

## 5. 最近实验现象与结论

最近真实 run：

```text
/mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2_agent_loop/v9p2d_01_courtyard_museum_agentloop_s1layout_noproxy_recall_contractprompt_max5_20260527_111458
```

9999 链接：

```text
http://192.168.192.123:9999/projects/buildingblock/runs/b3Blbl9zZW1hbnRpY19zMl9hZ2VudF9sb29wL3Y5cDJkXzAxX2NvdXJ0eWFyZF9tdXNldW1fYWdlbnRsb29wX3MxbGF5b3V0X25vcHJveHlfcmVjYWxsX2NvbnRyYWN0cHJvbXB0X21heDVfMjAyNjA1MjdfMTExNDU4/
```

Trace 摘要：

```json
{
  "status": "complete",
  "accepted": false,
  "stop_condition": "current_part_stage_max_iterations_exhausted",
  "final_scene_score": 0.5294,
  "iterations": 10
}
```

重要现象：

- T2I 审查/重生成/复审机制已经更接近正确工作流；
- 但 mesh / I23D 仍是当前主要 blocker；
- 对高墙 / gallery bar 这种 bbox 显示应为高且长的结构，T2I 有时会被 accept，但 I23D 生成结果仍会变成矮墙、柱、T-shaped、轴向旋转错误或塌缩；
- 这说明 **仅靠 VLM 判断 image accept 不足以保证 I23D 成功**，T2I prompt 必须带有 I23D-friendly geometry prior，mesh 阶段还需要 deterministic axis/pose contract repair。

## 6. 重要设计决策

### 6.1 不引入 bbox proxy 图作为 VLM 输入

用户不希望 `part_geometry_contract_bbox_proxy · display-enhanced from dark render` 成为主要输入，因为它不像真实 Stage1/S2 可视化逻辑，容易引入额外概念。

当前策略：

- layout context 回到 S1-style multiview；
- 只突出当前 layout part；
- 不再展示 bbox proxy；
- focused cards 展示真实 VLM 输入图，不用 display-enhanced 替换证据图。

### 6.2 可视化必须保留中间变量

之前 old/new T2I image 一样，根因之一是文件覆盖/引用方式不利于前后对比。

正确机制：

- 每次 Agent action 生成 snapshot；
- old image、new image、old mesh、new mesh 都应指向不同 artifact；
- prompt、metadata、stderr_tail 也随 action 固化；
- 不要只引用当前 latest 文件；
- 未来支持 rollback 到 N 次之前。

### 6.3 stage max iteration 要按阶段清零

`max=5` 应该约束当前阶段，例如 T2I 阶段最多 5 次；T2I accept 后进入 mesh 阶段，计数应清零。否则前一阶段耗尽次数会导致后一阶段提前停止。

## 7. 如何运行 / 恢复实验

### 7.1 推荐真实 run 模式

用户偏好：不要 dry-run / mini-run，除非明确说 smoke test。

常用命令形状：

```bash
cd /home/wangyz/project/0working/BuildingBlock/BuildingBlock
python scripts/run_archstudio_s2_agent_loop.py \
  --run-dir /mnt/data/wangyz/BuildingBlock/outputs/open_semantic_s2/v9p2d_01_courtyard_museum_partfocus_v10 \
  --agent-scope assemble_geometry \
  --evaluator-provider gpt55 \
  --evaluator-mode focused \
  --max-iterations 5 \
  --t2i-gpu 0,1,2,3,6,7 \
  --hunyuan-gpu 6,7 \
  --fast-report-refresh
```

实际参数以 `scripts/run_archstudio_s2_agent_loop.py --help` 为准。

### 7.2 刷新 9999 报告

```bash
cd /home/wangyz/project/0working/BuildingBlock/BuildingBlock
python - <<'PY'
from pathlib import Path
from scene_synthesis.building_mesh.report_bundle import refresh_s2_9999_report
run = Path('/tmp/s2_seqgate_latest_run.txt').read_text().strip()
print(refresh_s2_9999_report(run, full=False, trigger='manual'))
PY
```

生成 9999 URL：

```bash
python - <<'PY'
import base64
from pathlib import Path
run = Path('/tmp/s2_seqgate_latest_run.txt').read_text().strip()
rel = 'open_semantic_s2_agent_loop/' + Path(run).name
slug = base64.urlsafe_b64encode(rel.encode()).decode().rstrip('=')
print('http://192.168.192.123:9999/projects/buildingblock/runs/' + slug + '/')
PY
```

### 7.3 快速检查 trace

```bash
python - <<'PY'
import json
from pathlib import Path
run = Path('/tmp/s2_seqgate_latest_run.txt').read_text().strip()
trace = json.loads((Path(run) / 'agent_loop' / 'agent_trace.json').read_text())
print(trace.get('status'), trace.get('accepted'), trace.get('stop_condition'))
for it in trace.get('iterations', []):
    ev = it.get('evaluator', {})
    print(it.get('iteration'), ev.get('current_part_stage'), it.get('num_active_findings'),
          [(op.get('operation_type'), op.get('status')) for op in it.get('operations', [])])
PY
```

## 8. 验证命令

当前最重要的轻量 regression tests：

```bash
cd /home/wangyz/project/0working/BuildingBlock/BuildingBlock
PYTHONDONTWRITEBYTECODE=1 python scripts/tests/test_archstudio_s2_quality_agent.py
PYTHONDONTWRITEBYTECODE=1 python scripts/tests/test_archstudio_s2_repair_agent.py
```

若改动 report / visualization，建议额外：

```bash
python -m py_compile \
  scene_synthesis/building_mesh/quality_agent.py \
  scene_synthesis/building_mesh/s2_repair_agent.py \
  scene_synthesis/building_mesh/report_bundle.py \
  scene_synthesis/building_mesh/visualization.py \
  scripts/run_archstudio_s2_agent_loop.py
```

## 9. 当前已知问题

1. **I23D mesh 质量和轴向问题仍是主 blocker**  
   常见现象：长墙被旋转到 depth、front/back 只看到薄边、生成成 T-shaped column/beam、或局部塌缩。

2. **T2I accept 标准还需要更强的 geometry prior**  
   VLM 可能认为图像语义正确，但对 I23D 来说图像不是好先验，例如高墙 bbox 却生成矮墙图。

3. **9999 内容容易过多**  
   Focused Visual Critic QA 是对的，但 workflow summary / raw JSON 需要继续折叠和分层，默认展示应该面向调试者能快速理解。

4. **old/new artifact snapshot 必须持续检查**  
   不能回到 old/new image 或 old/new mesh 指向同一个 latest 文件的问题。

5. **VLM provider 偶发 error 要可审计**  
   失败时要显示 provider error、stderr_tail、request path，而不是吞掉或跳过。

## 10. 下一步规划

### 10.1 Mesh axis / pose contract repair

当前最值得做的系统性方法：

- 对生成 mesh 计算 bbox extents；
- 与 layout bbox 的 x/y/z extents 对齐；
- 枚举候选变换：identity、yaw 90/180/270、必要时 mirror；
- 对每个候选归一化到 layout bbox；
- 用 heuristic 或 VLM score 判断哪个最符合 long-axis / shallow-depth / z-up contract；
- 选中后作为 `Agent action · mesh_axis_alignment` 显示在 9999；
- 展示 before/after six-view mesh render；
- 加测试：x/y swapped synthetic wall 能被纠正，已对齐 mesh 不被改变。

### 10.2 进一步收紧 T2I critic

让 T2I 审查不仅问“图像是否符合文本”，还问：

- 这张图是否是 **I23D-friendly single object view**；
- 是否与 bbox shape family 强一致；
- 是否会诱导 Hunyuan 生成地面、台阶、门洞、柱、整栋建筑；
- 如果 bbox 是 tall long thin wall，reference image 必须明显是高长薄墙，而不是矮 parapet。

### 10.3 Assemble agent 回归到真实完整 run

等 per-part T2I / mesh 更稳定后，再跑完整 assemble_geometry：

- per-part 全部 accept 后 assemble；
- 整体 colored mesh six-view；
- VLM 输出必须具体到 part label；
- 修复可以 route 到 part-level T2I / I23D，而不是全局重跑。

### 10.4 9999 信息架构

建议后续页面结构：

1. Run overview：status、stop condition、score、9999 refresh time；
2. Current part timeline：T2I loop → mesh loop → accepted；
3. Agent action cards：T2I generation / geometry generation / mesh alignment；
4. VLM call cards：输入图、核心 prompt、answer summary、JSON link；
5. Collapsed raw trace：默认折叠。

## 11. Git / 数据注意事项

不要提交：

- `/mnt/data/...` outputs；
- `BuildingBlock/outputs`；
- BoxCenterSizeLabel 数据目录；
- checkpoints；
- `__pycache__` / `*.pyc`；
- `.omx/`；
- core dump；
- 本机路径 token / credential。

应该提交：

- Agent 相关 Python 源码；
- tests；
- docs；
- `.gitignore` 对本地生成物的过滤更新。

## 12. 恢复工作时的第一步 checklist

1. `git pull` 后先看本文；
2. 跑两个 regression tests；
3. 打开最近 9999 URL，确认页面仍可访问；
4. 用真实 run 跑一个 part，确认 T2I revise → regenerate → re-review 没退化；
5. 优先实现 mesh axis / pose contract repair；
6. 每次实验结束都给出 9999 链接。
