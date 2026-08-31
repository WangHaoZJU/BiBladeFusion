# 未知叶片今晚完整真机运行清单

适用目标：人工三视角启动 → 粗扫 → schema-5 → 精扫 → 最终模型。

本清单按实验模式编写。实验模式保留全部运动安全门，但跳过尚未完成的科学与运行时发布验收，
最终结果只能作为实验数据，不能声明为生产验收结果。

## 0. 三个不可违反的身份规则

- `placement_id`表示夹具和叶片一次不变的真实物理放置。
- `run_id`表示该放置下的一次软件运行或重试。
- `output`表示该次运行的独立、不可覆盖实验目录。

今晚第一次放置可使用：

```text
placement_id = blade-placement-20260831-01
run_id       = blade-placement-20260831-01-attempt-01
output       = data/experiments/blade-placement-20260831-01-attempt-01
```

同一物理位置的软件失败重试：保留`placement_id`，把attempt改为02并使用全新output。夹具或叶片
发生任何移动：停止当前实验，创建`blade-placement-20260831-02`，旧地图、粗模型、schema-5、
精扫覆盖和permit全部不得复用。

## 1. 放置前安全边界

- [ ] 现场至少一人始终能直接触及物理急停。
- [ ] 机器人速度缩放保持local配置中的`0.08`，不临时提高。
- [ ] D435i、末端连接件、机器人底座和工作台均未移动。
- [ ] 夹具和叶片在整个实验中刚性固定，实验开始后不再触碰。
- [ ] 整个叶片和夹具完全位于静态自由验收留下的目标包络内：

```text
x = 0.47 ～ 0.63 m
y = -0.10 ～ 0.14 m
z = 0.00 ～ 0.37 m
```

- [ ] 夹具本体仍位于配置的保守碰撞AABB内：

```text
x = 0.48 ～ 0.62 m
y = -0.095 ～ 0.135 m
z = -0.02 ～ 0.06 m
```

任一项不成立，不运行运动入口。重新测量并更新验收资产，不能仅凭“移动很小”继续。

## 2. eiai代码和环境准备

在项目根目录执行：

```bash
git status --short
git pull --ff-only origin main
git status --short
git log -1 --oneline
```

- [ ] 第一次`git status`没有不明修改；如有修改先人工确认归属。
- [ ] pull成功且没有合并提交。
- [ ] pull后的`git status`为空。
- [ ] 最新提交包含正式`placement_id`功能。

安装/核对目标依赖：

```bash
uv sync --extra foundation-stereo --extra tsdf-open3d
uv run python -c "import open3d; print(open3d.__version__)"
uv run bbf camera list
uv run bbf stereo doctor --config configs/local.yaml
```

- [ ] Open3D可以导入。
- [ ] 只出现目标D435i序列号`243222074585`。
- [ ] FoundationStereo源码、配置、权重、Torch和CUDA检查通过。

## 3. local配置核对

不得覆盖Git中的`configs/default.yaml`。今晚只使用eiai的`configs/local.yaml`。

确认以下值：

```yaml
robot:
  robot_ip: 192.168.6.60
  local_ip: 192.168.6.61
  motion_enabled: true

realsense:
  serial_number: "243222074585"
  infrared_emitter_enabled: true

stop_and_capture:
  enabled: true
  maximum_segment_joint_delta_rad: 0.02

occupancy:
  enabled: true
  integration_stride: 2
  maximum_map_age_s: null
  minimum_source_views: 3
  maximum_source_views: 3
  minimum_free_observations: 3
```

运行解析检查：

```bash
uv run python -c "from biblade_fusion.core.settings import load_settings; s=load_settings('configs/local.yaml'); print(s.occupancy.maximum_map_age_s, s.occupancy.minimum_source_views, s.occupancy.maximum_source_views, s.occupancy.minimum_free_observations)"
```

必须输出：

```text
None 3 3 3
```

禁止在实验开始后修改local配置、标定、模型、STL、FoundationStereo权重或多边形。

## 4. 新放置的第一帧预览与多边形

### 4.1 将机器人放到第一安全视角

- [ ] 用示教器把相机放到已知安全、能看到完整可见叶片和鳍片的正面视角。
- [ ] 确认电缆无拉扯、镜头无遮挡、叶片没有超出图像边界。
- [ ] 停止机器人程序，使控制器最终报告`robot_mode=IDLE`。
- [ ] 从此刻到正式第一帧采集完成，不移动机器人、相机、夹具或叶片。

### 4.2 预采集一份同步session

```bash
uv run bbf acquire snapshot \
  --config configs/local.yaml \
  --view-id blade-placement-20260831-01-preview
```

记录命令打印的session绝对路径。下面用`<PREVIEW_SESSION>`表示该路径。

### 4.3 对预览session运行FoundationStereo

输出目录必须不存在：

```bash
uv run bbf stereo infer-session \
  --config configs/local.yaml \
  --session <PREVIEW_SESSION> \
  --view-id blade-placement-20260831-01-preview \
  --output data/placements/blade-placement-20260831-01/preview-stereo
```

导出整流左图用于标注：

```bash
uv run python -c "import cv2, numpy as np; a=np.load('data/placements/blade-placement-20260831-01/preview-stereo/left_rectified.npy'); assert cv2.imwrite('data/placements/blade-placement-20260831-01/left_rectified.png', a)"
```

### 4.4 手动画完整`hard_roi`

在1280×720的`left_rectified.png`上记录多边形顶点，按轮廓顺序写入：

```text
data/placements/blade-placement-20260831-01/bootstrap_polygon.json
```

格式：

```json
{
  "vertices_uv": [
    [u0, v0],
    [u1, v1],
    [u2, v2]
  ]
}
```

- [ ] 顶点全部位于`0≤u<1280, 0≤v<720`。
- [ ] 多边形不自交，沿顺时针或逆时针连续排列。
- [ ] 包含全部可见主表面、鳍片、叶根、叶尖、前缘和后缘。
- [ ] 排除夹具、支架、机器人和背景。
- [ ] 不使用005的小表面seed作为完整hard ROI。

### 4.5 离线验证多边形

输出目录必须不存在：

```bash
uv run bbf scan bootstrap-mask \
  --config configs/local.yaml \
  --stereo data/placements/blade-placement-20260831-01/preview-stereo \
  --polygon data/placements/blade-placement-20260831-01/bootstrap_polygon.json \
  --seed-mode hard_roi \
  --output data/placements/blade-placement-20260831-01/bootstrap-mask-check
```

- [ ] 命令成功并打印`mask_pixels`和`mask_fraction`。
- [ ] 人工查看`mask.npy`或叠加图，确认完整叶片被保留且夹具未进入。
- [ ] 若失败或掩模错误：修改多边形，并使用新的检查输出目录；不得覆盖旧检查资产。
- [ ] 验证完成后机器人和实物仍未移动。

## 5. 非运动完整doctor

本次配置的科学和运行时发布验收为null，所以必须明确使用实验审计：

```bash
uv run bbf scan doctor \
  --mode unknown \
  --experimental \
  --config configs/local.yaml
```

- [ ] 命令返回0。
- [ ] 无FAIL项目。
- [ ] 明确理解实验结果不具有生产科学验收身份。

doctor失败时不启动机器人。保存完整输出，修复根因后重新运行doctor；不能只删检查或放宽阈值。

## 6. 正式运行前最后检查

- [ ] `data/experiments/blade-placement-20260831-01-attempt-01`不存在。
- [ ] 多边形文件已保存且不会再修改。
- [ ] 机器人仍在与预览相同的第一视角。
- [ ] 控制器为IDLE，安全状态正常，急停可触及。
- [ ] 没有人位于机器人工作空间内。
- [ ] 另一个终端已准备好，只用于只读监督。

正式命令：

```bash
uv run bbf scan run-unknown \
  --experimental \
  --config configs/local.yaml \
  --placement-id blade-placement-20260831-01 \
  --run-id blade-placement-20260831-01-attempt-01 \
  --output data/experiments/blade-placement-20260831-01-attempt-01 \
  --operator-id vale \
  --bootstrap-polygon data/placements/blade-placement-20260831-01/bootstrap_polygon.json \
  --bootstrap-seed-mode hard_roi \
  --first-side front
```

启动阶段会在连接硬件前预加载FoundationStereo。等待命令明确进入控制台，不要重复启动第二实例。

只读监督终端：

```bash
uv run bbf supervise replay \
  --snapshot data/experiments/blade-placement-20260831-01-attempt-01/live_timeline \
  --follow
```

## 7. 人工三视角启动

### 第一视角

控制台显示`NEEDS_CAPTURE`且没有预定view ID时：

- [ ] 再次确认控制器IDLE。
- [ ] 输入一次且仅一次：`c`。
- [ ] 等待stop、停稳、采集、FoundationStereo、自遮罩、占用重建和科学资产全部返回。
- [ ] 不因计算时间较长而再次输入`c`、移动机器人或关闭终端。

第一帧会使用命令绑定的hard ROI及`front`侧标记。地图通常仍为`MAPPING`。

### 第二视角

- [ ] 确认控制台重新提示人工重定位。
- [ ] 使用示教器把已停止机器人移动到第二个已知安全视角。
- [ ] 相机光心相对所有已接受视角至少平移2厘米，或观察方向至少变化5度。
- [ ] 保持叶片和夹具完全不动。
- [ ] 结束示教运动并确认控制器IDLE。
- [ ] 输入一次`c`，等待完整周期完成。

### 第三视角

重复第二视角流程，第三视角也必须与前两个视角分别满足独立性。建议三个视角具有明显不同的
相机方向，且至少为粗模型提供两侧证据；不能只在几乎相同姿态连续拍三次。

- [ ] 第三次输入`c`后等待地图达到`MAP_READY`。
- [ ] 若仍为`MAPPING`，查看阻塞原因；不得把minimum值改小或把UNKNOWN改成FREE。

安全地图只保留最近三个独立来源，但粗扫和精扫科学覆盖会持续累计。当前MAP_READY地图不会按
墙钟时间过期；只有下一次成功发布的新generation会替换它。

## 8. 自动粗扫与逐段批准

MAP_READY后，系统自动创建代理模型、普通法向候选和每侧±15°鳍片发现候选。操作员不手工指定
科学view ID。

每次控制台显示：

```text
Prepared one segment. Exact approval token: EXECUTE ...
```

逐段执行：

- [ ] 观察只读GUI中的目标、计划短段、机器人和占用图。
- [ ] 确认现场没有新增人员或障碍物，电缆状态正常。
- [ ] 确认计划运动方向符合预期。
- [ ] 仅在确认安全后，原样复制完整`EXECUTE ...`字符串并回车。
- [ ] 不手工缩短、重写或复用上一次token。
- [ ] 保持手在物理急停附近，观察整个短段。
- [ ] 每段结束后等待自动stop、停稳和采集完成。

较长路径会拆成多个`TRANSIT`段。TRANSIT采集只更新安全地图，不增加科学覆盖；系统会保持同一
最终科学目标。不要因为连续出现多个相似目标而中止或手工换目标。

粗扫不会在“看起来差不多”时结束。必须同时满足：

- [ ] 总粗扫视图数至少6。
- [ ] 正反侧各至少3。
- [ ] 每侧至少一对验证过的相反斜视图。
- [ ] 代理表面覆盖完成。
- [ ] 两侧鳍片都具有两个物理表面证据。

## 9. schema-5交接

粗扫门槛全部通过后系统自动：

1. 固化最终粗扫checkpoint；
2. 构建并严格重读schema-5；
3. 记录`handoff_prepared`；
4. 停止粗扫协调器；
5. 创建独立精扫协调器、占用publisher和安全工厂；
6. 写入精扫启动事件；
7. 在当前停止位姿执行`fine_transition_bootstrap_000`。

- [ ] 交接期间不移动机器人或叶片。
- [ ] 不启动第二个程序。
- [ ] 不把长时间计算误认为死机；观察日志和GPU利用率。
- [ ] 精扫若提示安全来源不足，只在明确提示时输入一次`c`，且不带front/back。

粗扫permit、批准、prepared segment和旧地图publication不会进入精扫；这是正常现象。

## 10. 精扫循环

精扫以schema-5为固定参考，逐项覆盖正反主表面、前后缘、叶根、叶尖、鳍片面、鳍根和自由边。

每个精扫目标仍执行与粗扫完全相同的短段批准流程：

- [ ] 检查GUI和现场。
- [ ] 原样粘贴当前唯一token。
- [ ] 观察短段并随时准备物理急停。
- [ ] 等待自动stop、停稳、采集、参考投影掩模、覆盖和质量评估完成。

同一patch质量不足时，系统只使用配置中三组确定性重采扰动，不允许无限尝试。没有候选通过
workspace、IK、FK或连续安全证明时会BLOCKED；不要现场放宽阈值继续运动。

## 11. 最终模型完成

只有全部必需patch通过覆盖率、RMSE、法向一致性和点数门槛后，系统才执行：

1. 完整精扫来源链重放；
2. 多视图融合；
3. 正反面独立TSDF；
4. 网格生成；
5. 两侧和鳍片完整性检查；
6. 边界边、边界环和watertight检查；
7. 写入最终不可变重建；
8. 写入`FINE_COMPLETED`。

成功标志：

- [ ] 控制台显示`Scan completed; no further motion is authorized.`
- [ ] 顶层链最后事件为`fine_completed`。
- [ ] 最终重建目录存在且可严格读取。
- [ ] GUI不再显示待批准运动。

实验模式的最终模型是可回放实验结果，但不会声称具有正式science acceptance。

## 12. 停止、急停和失败处理

### 正常主动停止

在控制台提示输入时键入`q`。等待程序确认stop并释放设备。不要直接关闭终端作为正常停止方式。

### 紧急情况

人员侵入、异常方向、碰撞风险、电缆拉扯或控制失常时，立即使用物理急停。软件`q`和Dashboard
stop不是安全等级急停替代品。急停后保存所有日志，不在原进程中继续。

### BLOCKED/FAILED

- [ ] 不删除失败目录。
- [ ] 不修改失败事件或资产。
- [ ] 保存控制台完整输出和`live_timeline`。
- [ ] 记录当时placement ID、run ID、最后动作和现场照片。
- [ ] 不移动实物时：新建`attempt-02`，沿用placement ID。
- [ ] 移动物体后：新建placement ID和attempt-01。

当前实验模式不允许`--resume`。失败后必须使用新run ID和新output；正式生产链完成验收后才使用
严格`--resume`。

常见阻塞处理：

| 现象 | 正确处理 |
|---|---|
| `robot_mode=RUNNING` | 停止控制器程序，确认IDLE后开始全新attempt |
| hard ROI失败 | 重画多边形，先离线bootstrap-mask验证 |
| 三帧后仍MAPPING | 检查独立视角、有效深度、自遮罩和FREE票，不降低门槛 |
| FoundationStereo失败 | 保存attempt，检查CUDA/权重/显存，不覆盖输出 |
| token mismatch | 没有运动；重新粘贴当前控制台打印的完整token |
| occupancy/continuous proof BLOCKED | 不运动；检查真实障碍、UNKNOWN和工作空间，不临时豁免 |
| schema-5缺鳍片双面 | 允许规划器继续粗扫；候选耗尽则停止分析数据 |
| 精扫候选耗尽 | 停止，分析固定参考、掩模、IK/FK和质量证据 |
| 最终网格门槛失败 | 保留精扫数据，离线分析；不能把失败改写成完成 |

## 13. 运行后数据核对与拷贝

读取正式放置身份和终态：

```bash
uv run python -c "from biblade_fusion.storage.unknown_blade_experiment import read_unknown_blade_experiment; s=read_unknown_blade_experiment('data/experiments/blade-placement-20260831-01-attempt-01/experiment_handoff'); print(s.placement_id, s.experiment_id, s.latest_event.event_type, len(s.events))"
```

预期前两项分别为placement ID和run ID；成功运行的最后事件为`fine_completed`。

- [ ] 保留整个实验根，不只复制最后模型。
- [ ] 同时保留本次local配置、标定、接受资产、模型和代码提交号的记录。
- [ ] 进程完全退出后再复制数据。
- [ ] 使用校验和或保留文件时间/权限的复制方式。

实验资产内部绑定eiai绝对路径。复制到Vale后可以分析数组和事件，但严格语义回放仍应在原eiai
路径与原运行环境完成；不能修改metadata中的路径来伪造可移植性。
