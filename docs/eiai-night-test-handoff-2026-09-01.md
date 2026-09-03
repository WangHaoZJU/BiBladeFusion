# eiai 真机实验接手文档（2026-09-01）

> 接手对象：eiai 电脑上的 Codex/开发代理与现场实验人员  
> 项目：BiBladeFusion  
> 仓库：`WangHaoZJU/BiBladeFusion`  
> 分支：`main`  
> 本次修复提交：`d7e19e2 Apply blade envelope across coarse reconstruction`  
> 实验目标：重新采集第一视角，完成至少三个人工启动视角，进入受监督粗扫，生成 schema-5，继续精扫和最终重建。

## 1. 接手结论

本次代码修改已经提交并推送到 `origin/main`。eiai 必须先拉取并确认当前提交为
`d7e19e2`，再配置本机 `configs/local.yaml`。不要只看到程序能够启动就开始运动；必须先完成
第一视角的 hard ROI 与 XYZ 包络离线回放，并让完整 unknown doctor 无任何 `FAIL`。

本次修改解决的是：第一视角或后续粗扫二维 mask 偏大时，桌面等非叶片点被带入粗模型的问题。
修复后的粗扫会对**每一帧** base 坐标系点云执行叶片 XYZ AABB 交集，只有交集内的支持点会进入
覆盖率、PCA、ICP、TSDF、曲面和鳍片分区。原始 hard-ROI 点云仍保留用于审计。

这不等于“真机必然成功”。AABB 只能排除盒外点，不能区分盒内的叶片和其他物体，也不能弥补
错误手眼标定、叶片移动、深度缺失、视角不可达、碰撞证明失败或最终网格不水密。代码的承诺是
在证据不足时 fail closed，不是自动放宽条件继续。

## 2. 绝对禁止事项

接手代理和现场人员不得为了让流程继续而执行以下操作：

- 不得修改或降低安全、覆盖率、质量、碰撞、独立视角和保留比例门限。
- 不得把 `UNKNOWN` 占据体素当作 `FREE`。
- 不得关闭机器人自遮罩、连续碰撞证明或操作员逐段批准。
- 不得直接修改 Git 跟踪的 `configs/default.yaml` 作为现场配置；只使用 eiai 的
  `configs/local.yaml`。
- 不得复用其他叶片放置的 polygon、XYZ 包络、占据地图、粗模型、schema-5 或精扫覆盖。
- 不得覆盖已有 preview、mask-check、initialization-check 或 experiment 输出目录。
- 不得在叶片、夹具或相机姿态改变后继续使用旧 `placement_id`。
- 不得用 `git reset --hard`、强制 checkout 或清理命令处理 eiai 上不明来源的本地修改。
- 不得把实验模式产生的结果声明为正式 science acceptance 结果。
- 人员侵入、异常运动、电缆拉扯或碰撞风险出现时，立即使用物理急停；软件 `q` 不是安全急停替代品。

## 3. eiai 拉取与代码身份核对

先在 eiai 的仓库根目录执行：

```bash
git status --short --branch
git pull --ff-only origin main
git log -1 --oneline --decorate
```

预期最后一条包含：

```text
d7e19e2 Apply blade envelope across coarse reconstruction
```

如果 pull 前 `git status --short` 显示不明修改，先停止。记录文件列表并让用户确认归属；不要覆盖。
如果 eiai 有自己的 `configs/local.yaml`，应保留它并在拉取后重新验证，而不是用 default 覆盖。

可进一步检查提交内容：

```bash
git show --stat --oneline d7e19e2
git status --short
```

拉取完成后的跟踪文件工作区应为空。本次提交不包含 eiai 的本地参数、模型权重或实验数据。

## 4. 2026-09-01 修改清单

### 4.1 修复初始化文件契约不一致

此前初始化 writer 实际生成 `metadata.json`，粗扫 consumer 却寻找 `initialization.json`，可能导致
第一帧已成功写盘但后续接收必然失败。

修复内容：

- 在 `src/biblade_fusion/storage/initialization.py` 定义共享常量
  `INITIALIZATION_METADATA_FILENAME = "metadata.json"`。
- 粗扫工作流和 coarse generation 统一使用该常量。
- 删除源代码、测试、README、文档中对 `initialization.json` 的旧引用。
- 测试改为使用真实 initialization writer 产生的 authority record，不再伪造文件名。

### 4.2 新增 base 坐标系叶片 XYZ 包络

新增文件：

```text
src/biblade_fusion/perception/proxy/support.py
tests/unit/perception/proxy/test_support.py
```

新增算法身份：

```text
base_frame_blade_envelope_aabb_v1
```

新增 `proxy_model` 配置：

```yaml
proxy_model:
  blade_envelope_min_m: [x_min, y_min, z_min]
  blade_envelope_max_m: [x_max, y_max, z_max]
  minimum_envelope_retained_fraction: <实测值>
```

三项必须同时存在或同时为空。unknown-blade 真机运行要求三项都存在。点云必须已经处于 `base`
坐标系，交集采用包含边界的比较：

```text
x_min <= x <= x_max
y_min <= y <= y_max
z_min <= z <= z_max
```

交集后同时检查：

- 保留点数不少于 `proxy_model.minimum_points`；
- 保留比例不低于 `minimum_envelope_retained_fraction`；
- 所有输入与诊断均有限、形状正确且能够重放。

任一条件失败都会终止当前科学资产，不会用少量交集点继续冒充完整叶片。

### 4.3 AABB 从“首帧代理”扩展到“每一帧完整粗扫链”

粗扫单帧存储 schema 从 1 升为 2。每个新 `coarse_scan_view` 都会保存：

```text
原始 reconstructed view
原始二维 foreground mask
proxy_support_mask.npy
proxy_model 配置
输入点数、保留点数、比例及输入/保留 XYZ 边界
```

严格 reader 会从原始点云和配置重新计算 support mask 与诊断，并拒绝任何漂移。

过滤后的 `support_cloud` 现在用于：

- 粗扫代理覆盖率更新；
- 粗扫 generation 重放；
- 最终粗模型 PCA；
- 正反面分类；
- 同侧有界 ICP；
- 厚度估计；
- 双侧 TSDF；
- 曲面分块；
- 两侧鳍片检测与区域划分。

原始 hard-ROI 点云不会被销毁。这样既能避免桌面污染模型，也能在实验后审计二维标注究竟包含了
哪些点。

### 4.4 离线重建与在线运行使用同一过滤逻辑

`bbf reconstruct coarse-model` 的离线路径也会对每个重建视图执行相同 support selection，再构建
粗模型。不得出现“在线有 AABB、离线重放没有 AABB”的两套结果。

### 4.5 schema-5 增加粗视图 support 来源证明

粗模型仍保持 schema-5，因为精扫入口明确要求精确 schema-5；没有为了本次修改随意升级精扫
契约。

schema-5 metadata 新增 `proxy_support` 证明，绑定：

- support 算法身份；
- 完整 `proxy_model` 配置；
- 精确 coarse-view 来源路径及 metadata 哈希；
- 来源顺序。

粗扫 generation 在 promotion 时会检查 schema-5 中的 source list 与 generation 的 coarse views
完全一致。恢复/复用已有粗模型时，如果缺少 support 来源、配置不同、来源顺序不同或哈希变化，
会拒绝复用。

### 4.6 初始化代理改为重新计算验证

初始化资产 reader 不再只相信保存的 proxy metadata。当配置启用 AABB 时，会使用保存的 support
点重新执行 bilateral proxy 构建，并比较：

- frame matrix；
- extents；
- centroid；
- PCA eigenvalues；
- 原始、有限和体素点数；
- 相机与法向余弦。

篡改 proxy extents 或点云后，严格读取会失败。

### 4.7 修复失败清理与同一动作重试

首帧初始化现在以事务方式处理 initialization、coarse-view、planning/discovery 资产。普通失败时会
清理本次新建的半成品，并且只有全部成功后才更新 session 状态。

如果 coarse generation append 在 coverage 已写入后失败，会删除未提交的 coverage 目录；同一个
append 可以安全重试，不会被孤立目录阻塞。

首帧 planning 资产已经成功但 generation append 失败时，重试会复用已封存的首帧资产，不会无故
重新制造另一套初始化结果。

### 4.8 后续自动前景仍保持 fail closed

2026-09-02的后续修复取代了本节最初的“无seed自动深度连通域”行为。第一帧formal capture仍使用
用户绑定的hard ROI；第二帧及以后把上一已验收generation中的累计叶片support点投影到当前图像，
膨胀后再与当前有效深度及base坐标系叶片专用AABB求交。它不复制上一视角二维polygon，也不在
夹具、桌面和叶片之间选择最大连通块。

投影证据不足、当前深度与叶片AABB不一致或MASK门限失败时仍会阻塞。自动MASK及其参考
`generation.json`、累计点哈希和配置写入schema-3粗扫视角，可独立重放；安全占用图继续使用全场景
有效深度，不受科学MASK裁剪。

### 4.9 文档与测试

更新：

- `README.md`
- `configs/default.yaml` 注释与 fail-closed 空默认值
- `docs/development-log.md`
- `docs/unknown-blade-night-run-checklist.md`

提交前验证结果：

```text
完整测试：1061 passed, 2 skipped
本次提交前定向回归：162 passed
ruff check .：通过
python -m compileall -q src tests：通过
git diff --check：通过
```

两个 skip 是既有的可选 Open3D renderer 测试；不是本次粗扫修复失败。

## 5. 修改涉及的核心文件

接手代理定位问题时优先查看：

| 文件 | 作用 |
|---|---|
| `src/biblade_fusion/perception/proxy/support.py` | AABB 交集、点数/比例门限、诊断 |
| `src/biblade_fusion/core/settings.py` | 新 proxy_model 配置及一致性验证 |
| `src/biblade_fusion/storage/initialization.py` | `metadata.json` 契约、proxy 重放 |
| `src/biblade_fusion/storage/coarse_scan.py` | schema-2 单帧 support 与 generation 验证 |
| `src/biblade_fusion/workflows/initialization.py` | 首帧 support selection |
| `src/biblade_fusion/workflows/unknown_blade_coarse.py` | 在线逐帧 support、事务清理、最终模型来源 |
| `src/biblade_fusion/workflows/coarse_model.py` | 最终建模消费过滤后的视图 |
| `src/biblade_fusion/storage/coarse_model.py` | schema-5 support provenance |
| `src/biblade_fusion/workflows/unknown_blade_runtime.py` | unknown runtime 配置准入 |
| `src/biblade_fusion/cli.py` | 离线 coarse-model 与 operator diagnostics |

## 6. 当前没有在代码仓库中完成的事项

以下内容必须在 eiai 和真实工作站完成，不能从 `configs/default.yaml` 猜测：

- 当前 placement 的 `blade_envelope_min_m`；
- 当前 placement 的 `blade_envelope_max_m`；
- 实测 `minimum_envelope_retained_fraction`；
- 叶片保守厚度 `estimated_thickness_m`；
- 精扫 standoff 基准、上下界；
- view-filter workspace；
- eiai 的 ES68 IK/model path；
- FoundationStereo 权重、配置、Torch/CUDA 环境；
- 占据 workspace、静态自由区域及验收绑定；
- motion-envelope acceptance 绑定；
- 真机 tracking、stop 和碰撞安全状态。

本次本地开发机的 `configs/default.yaml` 有意保持这些值为 `null`/disabled，因此用 default 运行
unknown doctor 必须失败。这是安全设计，不是需要删除的错误。

## 7. eiai 环境与依赖准备

建议依次执行：

普通 `git pull` 后不要仅为代码更新执行 `uv sync`。若环境确实需要重建，使用仓库脚本并传入
eiai 上实际的 Elite SDK wheel；脚本会在所有 locked extras 之后最后安装该私有 wheel：

```bash
./scripts/bootstrap-gpu.sh \
  /absolute/path/to/elite_cs_sdk-1.0.0-cp312-cp312-linux_x86_64.whl \
  configs/local.yaml
/usr/bin/env -u PYTHONPATH .venv/bin/python -c "import open3d; print(open3d.__version__)"
/usr/bin/env -u PYTHONPATH .venv/bin/bbf camera list
/usr/bin/env -u PYTHONPATH .venv/bin/bbf stereo doctor --config configs/local.yaml
```

确认：

- FoundationStereo repository、checkpoint、cfg、Torch、CUDA 全部通过；
- Open3D 可用或明确接受确定性后备路径；
- 只连接目标 D435i，序列号与 local 配置一致；
- stereo calibration 和 flange-primary hand-eye asset 可严格读取；
- 机器人、相机、底座、末端安装和工作台相对标定状态未改变。

如果安装依赖或模型失败，不启动运动入口。保存完整 doctor 输出后处理根因。

## 8. `configs/local.yaml` 参数配置顺序

不要一次填完后直接运动。建议按以下顺序逐层验证。

### 8.1 设备与标定

核对：

```yaml
robot:
  robot_ip: <eiai 实际值>
  local_ip: <eiai 实际值>
  motion_enabled: true

realsense:
  serial_number: <目标 D435i>
  infrared_emitter_enabled: true
```

确认 calibration 路径指向当前有效 stereo 与 hand-eye 资产。

### 8.2 叶片几何

```yaml
proxy_model:
  estimated_thickness_m: <实测或保守值>
  blade_envelope_min_m: [<x_min>, <y_min>, <z_min>]
  blade_envelope_max_m: [<x_max>, <y_max>, <z_max>]
  minimum_envelope_retained_fraction: <重复回放得到的门限>
```

AABB 必须以 `base` 表达，覆盖叶根、叶尖、前后缘、主表面和两侧凸出鳍片，并计入深度噪声、
手眼误差、机器人定位误差和本次固定放置误差。margin 不能大到重新包含整片桌面。

`minimum_envelope_retained_fraction` 是 `AABB 内点数 / 当前二维 mask 重建点数`，必须从正确
hard ROI 的实测回放得到。不能直接设为 0，也不能照搬其他 placement。

### 8.3 精扫科学参数

至少核对：

```yaml
blade_foreground:
  enabled: true

view_planning:
  standoff_distance_m: <实测基准>
  minimum_standoff_distance_m: <实测下界>
  maximum_standoff_distance_m: <实测上界>

view_filter:
  workspace: <验收过的范围>

kinematics:
  model_path: <eiai 实际 ES68 模型>
```

不要现场随意修改精扫前后深度容差、投影 radius 或质量门限来追求通过。

### 8.4 运动与占据安全

至少核对：

```yaml
stop_and_capture:
  enabled: true
  maximum_segment_joint_delta_rad: 0.02
  require_operator_approval: true
  require_capture_after_every_segment: true

occupancy:
  enabled: true
  minimum_source_views: 3
  maximum_source_views: 3
  minimum_free_observations: 3
  workspace_bounds_min_m: <实测>
  workspace_bounds_max_m: <实测>
```

同时绑定当前机器人几何对应的 static-free acceptance 与 motion-envelope acceptance。不要通过删除
accepted-static-free 配置来绕过 UNKNOWN；缺失证明就应阻塞。

## 9. 第一视角预览、hard ROI 与 AABB 离线验收

第一视角预览不属于正式三视角账本。其用途是为当前相机姿态和当前 placement 生成新的 polygon
与包络检查资产。

### 9.1 预览采集

机器人由示教器放到第一安全视角，确认控制器最终为 IDLE，叶片完整入镜。随后执行：

```bash
uv run bbf acquire snapshot \
  --config configs/local.yaml \
  --view-id <placement-id>-preview
```

记录命令返回的 `<PREVIEW_SESSION>`。从预览开始到正式第一帧完成，机器人、相机、夹具和叶片都
不得移动。

### 9.2 FoundationStereo 与 polygon

```bash
uv run bbf stereo infer-session \
  --config configs/local.yaml \
  --session <PREVIEW_SESSION> \
  --view-id <placement-id>-preview \
  --output data/placements/<placement-id>/preview-stereo
```

在该输出的 `left_rectified.npy` 对应图像上重新标注完整 hard ROI。polygon 使用整流左图像素坐标，
不能复用不同相机姿态的旧 polygon。

离线验证：

```bash
uv run bbf scan bootstrap-mask \
  --config configs/local.yaml \
  --stereo data/placements/<placement-id>/preview-stereo \
  --polygon data/placements/<placement-id>/bootstrap_polygon.json \
  --seed-mode hard_roi \
  --output data/placements/<placement-id>/bootstrap-mask-check
```

人工检查 mask/叠加图：必须覆盖完整可见叶片和鳍片，尽量排除桌面、夹具、支架、机器人和背景。
hard ROI 内所有有效深度都会被保留，它不会在 ROI 内再自动猜哪一部分是叶片。

### 9.3 AABB 初始化回放

```bash
uv run bbf initialize stereo-depth \
  --config configs/local.yaml \
  --session <PREVIEW_SESSION> \
  --stereo data/placements/<placement-id>/preview-stereo \
  --view-id <placement-id>-preview \
  --mask data/placements/<placement-id>/bootstrap-mask-check/mask.npy \
  --output data/placements/<placement-id>/initialization-envelope-check
```

必须记录和核对：

- `Proxy support: retained N/M hard-ROI points`；
- `N/M` 与实测 `minimum_envelope_retained_fraction` 的关系；
- 原始 `base_points_m.npy` 点数为 M；
- `proxy_support_mask.npy` 保留点数为 N；
- retained XYZ 没有裁掉叶尖、叶根、边缘或鳍片；
- 桌面和夹具点没有残留到 support overlay；
- metadata 中的输入/保留 bounds 可以从数组重新计算。

只验证 `Z > 0` 不足以去除桌面。此前本地样本中 `Z > 0` 仍保留约 99.88% ROI 点，因此不能把
它当作叶片分割策略。

## 10. 完整非运动 doctor

今晚若使用实验模式，执行：

```bash
uv run bbf scan doctor \
  --mode unknown \
  --experimental \
  --config configs/local.yaml
```

准入条件：退出码为 0，且没有任何 `FAIL`。

`--experimental` 只跳过正式 science acceptance 与 runtime-timing release 声明；它不会跳过：

- 相机与机器人配置；
- FoundationStereo/CUDA；
- 标定；
- AABB、厚度、standoff、workspace 与 IK/FK；
- 三态占据；
- 机器人自遮罩；
- continuous mesh/occupancy sweep；
- motion-envelope acceptance；
- stop、停稳和逐段人工批准。

如果 doctor 失败，接手代理应按项目名和 `details.missing` 修复根因并重新运行。不得通过删除检查
或改成 warning 放行。

## 11. 正式实验身份与启动命令

身份规则：

- `placement_id`：叶片和夹具一次不变的物理放置；
- `run_id`：该 placement 下的一次软件 attempt；
- `output`：该 attempt 的全新、不可覆盖目录。

示例：

```bash
uv run bbf scan run-unknown \
  --experimental \
  --config configs/local.yaml \
  --placement-id <placement-id> \
  --run-id <placement-id>-attempt-01 \
  --output data/experiments/<placement-id>-attempt-01 \
  --operator-id vale \
  --bootstrap-polygon data/placements/<placement-id>/bootstrap_polygon.json \
  --bootstrap-seed-mode hard_roi \
  --first-side front
```

正式启动前确认 output 不存在、控制器 IDLE、急停可触及、工作空间无人、现场障碍物与验收场景
一致。启动阶段会先预加载 FoundationStereo，再接触运动硬件。

## 12. “重新采集第一视角 + 人工三视角 + 粗扫”的精确定义

总流程是：

```text
一个离线 preview 帧（不进入正式账本）
    ↓
正式第一帧：在同一姿态重新采集，使用 hard ROI，计为人工视角 1
    ↓
示教器移动到独立安全姿态，正式采集人工视角 2
    ↓
示教器移动到另一独立安全姿态，正式采集人工视角 3
    ↓
MAP_READY 后进入受监督自动粗扫
```

因此不是“preview + 第一帧 + 另外三帧”，而是 preview 后重新采集的正式第一帧属于至少三帧之一。

第一帧 formal capture 使用 polygon hard ROI 和 `front`。第二、第三帧不会复用第一帧 polygon，
通常使用自动深度连通域，然后经过本次新增的 XYZ AABB。

人工视角之间至少满足以下一个独立性条件：

```text
相机光心平移 >= 0.02 m
或观察方向变化 >= 5 deg
```

第三帧后如果仍为 `MAPPING`，查看独立性、深度、自遮罩和 FREE 投票诊断。允许增加人工视角；
不允许降低 `minimum_source_views` 或把 UNKNOWN 改成 FREE。

## 13. 受监督粗扫阶段的观察点

MAP_READY 后，系统自动选择粗扫候选，但每个运动短段仍需要当前唯一批准 token。长路径会拆成多个
`TRANSIT` 段；TRANSIT 只更新安全地图，不增加科学覆盖。

每个接受的 coarse view 应检查或记录：

- `coarse_scan_view` schema 为 2；
- 存在 `proxy_support_mask.npy`；
- metadata 中 AABB 配置与当前 local 配置一致；
- retained count/fraction 合理；
- retained bounds 未裁掉叶片；
- 当前 view 的 support overlay 没有桌面/夹具明显残留；
- coverage generation 精确引用当前 coarse view；
- 不存在被上一次失败遗留的孤立 coverage 目录阻塞重试。

粗扫默认不能仅凭视觉“看起来完整”结束。通常至少需要：

- 总视图数不少于 6；
- 正反侧各不少于 3；
- 每侧有相反斜视发现证据；
- 代理覆盖完成；
- 两侧鳍片具有双面证据。

如果自动连通域 mask 选错或失败：停止并保存当前 attempt。不要复用第一帧 polygon 到另一个视角，
也不要临时扩大 AABB。首先检查该视角是否完整入镜、是否触边、深度是否连续、是否有遮挡以及
support retained diagnostics。

## 14. schema-5 交接验收

粗扫完成后应观察：

- generation 封存成功；
- schema-5 metadata 严格重读成功；
- `proxy_support` 算法、配置和 coarse source records 完整；
- schema-5 source coarse-view 顺序与 generation 完全一致；
- handoff event 已写入顶层事件链；
- 精扫使用同一个 schema-5 metadata SHA-256；
- 交接期间机器人和叶片未移动。

出现 schema-5 目录但没有对应 handoff/fine-start event 时，不得人工拼接目录宣布精扫已经开始。

## 15. 精扫流程和与 XYZ AABB 的边界

精扫以 schema-5 曲面为固定参考：

```text
未完成 patch
→ 候选视角/重采视角
→ workspace、投影、可见性、IK、FK
→ 当前占据地图和连续碰撞预检
→ 逐段批准、运动、停稳、FoundationStereo
→ schema-5 投影 z-buffer mask
→ 前后深度一致性
→ schema-3 精扫点云
→ immutable coverage successor
→ patch 点数/覆盖率/RMSE/法向门限
```

当前默认精扫 mask 的深度一致性起点为前方 6 mm、后方 10 mm；所有实际值以冻结的
`configs/local.yaml` 为准，现场不要随意调大。

重要边界：本次 XYZ AABB **直接硬裁剪的是粗扫每一帧**。精扫当前主要依靠干净 schema-5 的投影
与深度一致性，不会在每个精扫 schema-3 点云后再次应用同一 AABB。因此：

- 粗模型干净、叶片不动、标定准确时，桌面通常不会通过精扫参考 mask；
- 投影边缘膨胀范围内、且与叶片预测深度非常接近的桌面点理论上仍可能成为少量边界点；
- 叶片在粗扫后移动会使整个固定参考失效；
- 如未来增加“fine AABB 第二道 gate”，应单独设计、测试和验收，不要在今晚现场临时修改。

精扫 patch 不合格时只允许配置中有限的确定性重采；耗尽后 BLOCKED 是正确结果，不能无限随机
移动。

## 16. 最终完成条件

全部精扫 patch 达标只会触发最终重建，还不等于成功。最终阶段会重放全部 foreground-bound
schema-3 视图，执行多视图融合、正反面独立 TSDF、网格生成和终端门限。

默认终端门限包括：

- 每个固定参考 patch 均通过质量门；
- 正反侧均有独立来源视图；
- 正反侧均有 mesh triangles；
- 两侧各有一个双面鳍片；
- fin face/root/free-edge 区域完整；
- boundary edge 数为 0；
- boundary loop 数为 0；
- mesh watertight。

覆盖完成但最终水密/拓扑门失败时，正确终态仍是失败。保留资产进行离线分析，不得把失败 metadata
改成完成。

## 17. 常见阻塞分流

| 现象 | 正确处理 |
|---|---|
| eiai pull 后不是 `d7e19e2` | 停止，检查分支/remote，不运行旧代码 |
| doctor 报 AABB 缺失 | 在 local 中同时填写 min/max/fraction，并先做初始化回放 |
| `retained_fraction` 太低 | 检查 polygon、AABB、base 变换和叶片是否移动，不直接降低门限 |
| support 裁掉叶片 | 重新测量当前 placement 包络并创建新检查资产 |
| support 仍含桌面 | 收紧实测包络/修正 ROI；盒内重叠点不能靠 AABB 区分 |
| 第一帧 mask 错 | 重画当前整流左图 polygon，使用新输出目录 |
| 第二/第三帧 mask 错 | 检查自动连通域、触边、深度、遮挡和视角，不复用第一帧 polygon |
| 三帧后仍 `MAPPING` | 检查视角独立性、自遮罩和 FREE 票；必要时增加人工视角 |
| FoundationStereo/CUDA 失败 | 保存 attempt，检查环境/权重/显存，不覆盖输出 |
| occupancy 或 continuous proof BLOCKED | 不运动，检查真实障碍、UNKNOWN、workspace 和验收绑定 |
| schema-5 缺鳍片双面 | 让粗扫规划器继续；候选耗尽则停止分析数据 |
| 精扫 mask 像素过少 | 检查固定参考、位姿、遮挡、深度容差证据，不直接放宽门限 |
| 精扫候选耗尽 | 保存 coverage lineage，分析 IK/FK、可见性和质量，不随机游走 |
| 最终 mesh 非水密 | 保存完整精扫数据，离线分析，不伪造完成 |

## 18. 失败后的身份与数据处理

失败时：

- 不删除失败 output；
- 保存控制台完整输出、`live_timeline`、现场照片和最后动作；
- 如果实物完全没动，可以沿用 `placement_id`，创建新的 `run_id` 和全新 output；
- 如果叶片、夹具、相机安装或标定关系移动，必须创建新的 `placement_id`，旧地图/模型全部作废；
- 物理急停后不要在原进程中继续；
- 当前实验模式不把失败中的 permit、地图 publication 或 in-flight segment 当作可恢复状态。

## 19. 接手代理必须持续记录的内容

建议在 eiai 另建一份本次 attempt 日志，至少记录：

```text
UTC/本地时间
git commit
configs/local.yaml 的 SHA-256 或冻结副本路径
placement_id / run_id / output
stereo、hand-eye、motion/static-free acceptance ID
FoundationStereo checkpoint/cfg 身份
AABB min/max/fraction 与测量来源
preview session / polygon / mask-check / initialization-check 路径
每个人工视角的 view ID、位姿独立性和 MAP 状态
每个 coarse view 的 raw/retained 点数与比例
schema-5 路径与 metadata SHA-256
每个精扫 patch 的完成原因或失败原因
最终 reconstruction 路径、artifact ID、终端 gate 报告
所有 BLOCKED/FAILED 的原始错误文本
```

接手代理在做任何代码修改、阈值修改或验收资产更新前，应先向用户说明：当前证据、拟修改内容、
为什么这不是绕过 fail-closed，以及修改后必须重跑哪些离线验证。今晚优先目标是验证已提交主线，
不是现场扩展新功能。

## 20. 现有详细参考文档

更完整的现场数学原理、逐段批准、占据地图、schema-5、精扫与最终模型检查见：

```text
docs/unknown-blade-night-run-checklist.md
docs/supervised-blade-experiment.md
docs/stop-and-capture-coordinator.md
docs/coverage-next-view-selector.md
docs/occupancy-motion-safety.md
docs/development-log.md
```

若本接手文档与代码行为不一致，以提交 `d7e19e2` 的严格 reader、doctor 输出和实际错误为准，
并记录差异；不要凭文档文字强行继续运动。
