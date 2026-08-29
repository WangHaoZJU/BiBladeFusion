# BiBladeFusion 项目工作流详解

> 本文档基于对 `README.md`、`docs/supervised-blade-experiment.md`、`docs/stop-and-capture-coordinator.md`、`docs/development-log.md` 及源码目录结构的阅读整理而成。所有命令名均与项目内实际 CLI 入口一一对应，便于回查。

---

## 0. 项目定位

**BiBladeFusion** 是一个面向**薄壁叶片（涡轮叶片类）**的**机器人引导式双目三维几何重建系统**：

- **机械臂**：Elite ES68 六轴（eye-in-hand 配置）
- **相机**：Intel RealSense D435i（仅使用红外双目 `infrared/1`）
- **深度**：FoundationStereo 推理（官方源码作为 Git 子模块）
- **核心能力**：双面 + 鳍片（protruding fins）的薄壁曲面重建；粗→精两阶段闭环；写一次 + 哈希链 + 物理源身份 + 连续保守安全证明
- **当前状态**：2026-08-29 收盘——软件预验收边界已闭合，真机物理验收仍待完成

---

## 1. Schema-5 是什么

代码库中存在多处对 `schema-5` 的引用，它是本项目里一个**特定层级的"粗扫叶片曲面模型"**。

### 1.1 核心含义

1. **粗扫模型的第 5 版 schema**（coarse-model schema 5），而不是数据表 schema。代码里用 `kind: "coarse_model_schema_5"` 标识这种资产（见 `src/biblade_fusion/storage/surface_coverage.py:1201`）。
2. **不可变的粗曲面参考**（immutable coarse reference）——一旦通过升级门控，就成为精扫科学分支**唯一允许使用**的参考曲面。精扫覆盖率账本、FoundationStereo 周期分支、next-view 选择都"必须"使用它（见 `configs/default.yaml:117`）。
3. **从粗扫代际（coarse generation）到 schema-5 是单向（one-way）提升**：不继承旧的 prepared segment、permit、approval、旧地图 publication、精扫覆盖率（见 `docs/supervised-blade-experiment.md:73`）。

### 1.2 升级门控（必须同时满足）

依据 `docs/supervised-blade-experiment.md:67` 与 `docs/development-log.md:55`：

- 正反面最小视图数（front/back view count）
- 代理块覆盖（proxy coverage）
- 每侧至少一组配对的斜视（opposing oblique pair）
- 读回校验完整（complete schema-5 reader 必须成功读回）
- 与精确重构视图绑定（binds the model to the exact reconstructed-view）

任一失败 → **fail-closed**，停在 `BUILDING_SCHEMA5` 阶段（`unknown_blade_experiment.py:32`）。

### 1.3 与 schema 4 的关系

按 `docs/development-log.md:538`：

- **Schema 4**：初始化阶段，记录 depth source / projection frame
- **Schema 5**：粗扫完成后存储与读回的曲面模型

二者是不同的 schema 层（schema 4 → schema 5），升级路径分离。

### 1.4 时序预算

为防止 schema-5 构建/交接无限制挂起：

- 配置项：`stop_and_capture.maximum_schema5_handoff_duration_s`
- 监控入口：`_require_schema5_handoff_budget`，超时让协调器 `request_stop("schema-5 handoff invalidated the coarse coordinator")` 并保持 `BLOCKED`
- 相关 trace：`schema5_handoff_trace`，需要至少 3 组唯一 trial（含 cold 与 warm）

### 1.5 一句话总结

> **Schema-5 = 经过多重视图与覆盖门控、读回校验、单向升级后被"钉死"的不可变粗扫叶片曲面模型**，作为精扫科学分支、coverage ledger 与 next-view 规划的唯一合法参考；其构建/交接受时序预算约束，失败必须 fail-closed。

---

## 2. 项目整体介绍

### 2.1 项目定位（一句话）

**BiBladeFusion** 是一个面向**薄壁叶片（如涡轮叶片）**的**机器人引导式双目三维几何重建系统**，由 **Elite ES68 六轴机械臂**搭载 **Intel RealSense D435i 红外双目相机**，配合 **FoundationStereo 深度推理**与**自研安全监督状态机**，实现从粗扫到精扫的全流程、双面、可审计的曲面重建，目前已完成**软件验收前状态**，仍待真机物理验收。

### 2.2 核心应用场景

**目标对象**：薄壁叶片（thin-walled blades）——典型为带前后缘（leading/trailing edge）、叶根/叶尖（root/tip）、以及正反两面**伸出鳍片（protruding fins）**的工业叶片。

**核心问题**：双面 + 薄壁 + 鳍片，单面扫描无法获取完整几何；工业 CT/蓝光扫描成本高、不可现场化；普通 RGB-D 受基线与红外散斑限制，精度不足。

**解决方案**：
- **eye-in-hand**：D435i 装在 ES68 末端，机器人带动相机绕叶片多视角采集
- **FoundationStereo**：高精度学习式立体匹配替代 RealSense 原生深度
- **主动安全监督**：每个轨迹必须连续无碰，全程可审计、可回放

### 2.3 技术栈

| 层次 | 选型 |
|---|---|
| 编程语言 | Python 3.12（`uv` 管理依赖） |
| 机械臂 | Elite ES68（六轴），HoloRobot 控制生命周期复制/适配 |
| 相机 | Intel RealSense D435i（仅使用红外双目 `infrared/1`） |
| 深度推理 | **FoundationStereo**（官方源码作为 Git 子模块） |
| 立体标定 | PySide6 GUI + ChArUco + Zhang 初始化 + 联合 BA，**禁用出厂 IR 内参/外参** |
| 手眼标定 | Park-Martin 初始化 + LM/BA 精炼，**flange-primary** 链 |
| 运动学 | 自带 ES68 709 姿态标定 FK（拷贝自 HoloRobot）+ KDL 离线 IK |
| 几何 | NumPy / Open3D（可选）/ 自实现 marching tetrahedra；TSDF 双面融合；4 边界 B 样条 + 等弧长 Coons 曲面 |
| 碰撞 | Pinocchio + hpp-fcl（自适应关节区间二分 + 中点 FCL + 串联链保守半径） |
| 监督/UI | PySide6（标定 GUI + Qt3D 模型查看 + supervision replay 只读控制台） |
| 持久化 | 全部资产**写一次（write-once）**、SHA-256 绑定、前向哈希链 |
| 测试 | `pytest`（600+ 用例）、`ruff`、bytecode、whitespace、lockfile 一致性 |

### 2.4 整体工作流（监督式闭环）

```
操作员把相机放在已知安全的初始可见位姿
  ↓ stop + 停稳证据
  ↓ 左右红外采集
  ↓ FoundationStereo → 视差 / 置信度 / 深度
  ↓ 初始叶片前景（自动唯一连通分量，歧义时人工 ROI）
  ↓ ≥3 个操作员引导的、几何独立的视角 → 三态占用图（MAPPING → MAP_READY）
  ↓ 生成 / 更新叶片科学模型 + 覆盖账本（coverage ledger）
  ↓ coverage-driven 选下一个视点（next-view selector）
  ↓ 仅截取一段有界短关节段（one bounded segment）
  ↓ 连续机器人网格扫掠证明 + 连续机器人-占用体素扫掠证明
  ↓ 精确到当前 preflight 哈希的人工批准（一段一签）
  ↓ Guarded ServoJ 执行 → 显式 stop → 停稳
  ↓ 新视点再次采集 → 循环，直到正反面/边界/鳍片分区完成
```

### 2.5 关键创新点

1. **双向几何科学（Coarse → Fine）**：代理平面粗扫 → 鳍片斜视对 → schema-5 单向提升 → 固定参考精扫
2. **三态占用 + 连续扫掠安全证明**：FREE/OCCUPIED/UNKNOWN，UNKNOWN 阻塞；每个运动段必须通过两项独立连续证明
3. **写一次 / 哈希链 / 可审计状态机**：顶层链 `INIT → COARSE_CHECKPOINT+ → PREPARED → FINE_STARTED → FINE_CHECKPOINT* → FINE_COMPLETED`
4. **严格的物理身份而非逻辑身份**：schema-7 用 session manifest + view metadata + 序号 + frame number；切 view_id 不算独立
5. **防回滚式粗→精切换**：schema-5 升级单向、不迁移任何旧状态
6. **时序预算与科学验收的不可绕过性**：四预算 trace 必由实测产生，不接受手填数字

### 2.6 当前状态（2026-08-29 收盘点）

**已完成并回归验证（软件）**：ES68+D435i 标定、双面代理 + 鳍片发现 + 4 边界 B 样条 + Coons 不规则域 + 双面 TSDF、schema-5 单向提升 + 固定参考精扫、三态占用 + 物理源身份、两条连续扫掠证明、写一次顶层链 + 精确链式 `--resume`、四预算运行时验收资产、科学验收资产、公开的 `bbf scan run-unknown` 监督式运动入口、只读 `bbf supervise replay --follow` 监督 GUI。

**仍未做（真机验收项目）**：

1. 在目标 GPU 上验证 FoundationStereo 真实推理时长与 schema-5 handoff 时长
2. 在场、低速、可急停的 ES68 上接受最终 STL/manifest / static-free AABB / 跟踪-stop 包络 / 连续证明行为 / 工作空间边界
3. 真实双面 + 鳍片数据集 + 可追溯的尺寸基准 → 写入 `science_acceptance.path/id`
4. 跑 `scan doctor --mode unknown` → 现场分段批准协议 → 记录时序证据
5. 热成像分支目前只是**禁用的接口占位**

### 2.7 风险与限制

- **不能仅靠软件绿灯判断系统可用**：开发日志反复强调 "passing software doctor is not hardware acceptance"
- **占用 map age**：从完整 rebuild 周期第一帧起算，**不是连续动态避障**
- **机器人自遮罩**：自身及背后永远 `UNKNOWN`，需要"已验收静态自由 AABB"才能局部豁免
- **FoundationStereo 是必需**：原生 RealSense 深度在监督式运行链中**不是备用后端**
- **碰撞装配依赖 manifest**：缺/错的最终 STL 会立即 fail-closed
- **没有任何自动连续采集**：每个采集都需操作员按一次 `c`

---

## 3. 工作流每一步详解

### A. 离线准备阶段（运行前必须完成）

#### A1. 标定（calibration）

##### A1.1 D435i IR 立体标定

**原理**：Factory IR intrinsics / extrinsics **不能用**——它是为散斑匹配优化的，跟 FoundationStereo 使用的 rectified-left 像素坐标系不一致。

**流程**：

1. PySide6 GUI 启动后**默认空闲**，点"开始"才连接相机
2. 用 ChArUco（14×9）板在多个位姿采集**未整流的 Y8 红外对**（左右两路 `infrared/1`）
3. 每帧原子写入一份"latest-frame mailbox"——慢的检测不会形成事件积压
4. 离线做 ChArUco 角点检测 → 独立 Zhang 初始化 → 联合 stereo bundle adjustment
5. **不直接覆盖**上次结果，写到唯一会话目录（含 SHA-256 manifest），通过验收后才原子发布到 `data/calibrations/d435i_ir_active.yaml`
6. 之后用 `stereo-validate-gui` 在**新位姿**上做 hold-out 验证（不重拟合）——任一指标失败都不发布

##### A1.2 ES68 ↔ D435i 手眼标定（flange-primary）

**原理**：TCP 由控制器报告，存在控制器侧累积误差；FK 由标定过的 MDH 链算出来，更可信。

**流程**：

1. `calibration hand-eye-gui` 启动后空闲，按 `C` 收一组同步位姿（关节 + 红外左右 + ChArUco 角点）
2. 采集足够位姿后，用 Park-Martin 初始化 + 关节空间 SE(3) LM/BA 精炼，求 `flange_T_left_ir`
3. **目标坐标系 `base_T_target` 固定**，不参与精炼
4. 求解完后必须再走一遍 hold-out 验证（至少 5 个新位姿），过则原子发布到 `data/calibrations/es68_left_ir_hand_eye_active.yaml`
5. 同步做控制器 RTSI TCP 校验（FK vs TCP，2 mm / 0.3° 容差）——这是**校验通道**，不是求解通道

##### A1.3 ES68 MDH 标定

**原理**：拷贝自 HoloRobot 的 709 姿态标定 FK，含六个 joint-zero offset。TCP 旋转向量用 vendor 约定（roll/pitch/yaw），与 IK 的 RPY 编码刻意区分。

#### A2. 写一次资产与配置防御性副本

每个组件在构造时**拷贝自己关心的子配置**（Pydantic 规范化值）存到内部 payload；构造时核对任一字段不一致即拒绝组装。防止"调用方事后修改原 AppSettings 却影响已装配组件"。

#### A3. 物理验收资产（离线记录，软件 doctor 的硬前置）

| 验收资产 | 命令 | 必填字段 |
|---|---|---|
| 静态自由 AABB | `bbf safety record-static-free-acceptance` | 操作员、工作站、UTC、机器人几何 hash、工作空间、精确 AABB |
| 运动包络（跟踪 / stop 漂移） | `bbf safety record-motion-envelope-acceptance` | 六轴跟踪偏差、stop 漂移、6 类停稳速度阈值 |
| 几何科学验收 | `bbf safety record-science-acceptance` | 距离/入射角包络、绑定 FoundationStereo 源码/权重/模型配置、Python/OS/CUDA/GPU 运行时身份 |
| 运行时四预算 | `bbf safety record-runtime-timing-acceptance` | 感知周期 / 重定位间隔 / 段执行 / schema-5+精扫交接 4 个最坏值，每项至少 cold+warm 共 3 组 trial |

每一份都用 `measure_runtime_timing_trace` 在回调前后读 monotonic clock 写 trace，**不接受手填秒数**。trace 同时固化测量实现、runtime contract、Linux boot 身份、measurement-session 身份。

#### A4. 非运动就绪审计（`bbf scan doctor --mode unknown`）

只读、不连硬件、不发运动指令。逐项校验：

- 标定文件（IR 立体、手眼、FK、TCP offset）路径与哈希
- 工作空间、碰撞几何、静态自由 AABB 验收
- FoundationStereo 子模块 + CUDA + 推理依赖
- 最终 ES68+D435i STL manifest 完整且已 ready
- 科学/运行时验收资产存在且 contract 匹配
- 短段关节上限 / 工作空间未填写时**直接 fail**——必须人为填写

任一项缺失 → `scan doctor` 返回非零，`scan run-unknown` 不会启动。

### B. 运行入口（`bbf scan run-unknown`）

```bash
uv run bbf scan run-unknown \
  --config configs/local.yaml \
  --output data/experiments/unknown_blade_001 \
  --operator-id vale
```

`output` 目录必须不存在；写入开始后**永不覆盖**。

---

#### B1. 操作员把相机放到已知安全的初始可见位姿

**目的**：从单帧图像无法证明"遮挡后是空的"，代码不允许自动绕到叶片背面。必须由人用已单独确认安全的方式把机器人摆到位。

**动作**：

1. 操作员用示教器 / Dashboard 移动 ES68 到目标位姿
2. 协调器自动向 ES68 Dashboard **先发 stop**（production 入口建立运动驱动前必须经过 Dashboard stop）
3. **有界离散停稳采样证据**（不是连续证明）开始记录：实际/目标关节速度、实际/目标 TCP 线速度与角速度、控制器状态、控制器安全状态、host 时间和 controller 时间
4. 停稳窗口同时检查：运行态、`IDLE`、安全状态 `NORMAL/REDUCED`、关节与 TCP 速度 ≤ 阈值、时间戳新鲜度、stop generation 与当前协调器一致
5. 任一通道缺失/过期/超阈值 → fail-closed，不进入采集

---

#### B2. 同步采集一组双目红外

**原理**：必须用红外双目 + FoundationStereo 出深度；RealSense 原生深度**不能作为本链的安全地图/科学重建后端**，仅可作为独立对比评估。

**流程**：

1. `SessionWriter` 创建新周期目录 `cycles/<sequence>_<view>/raw/<single-view-session>/`，写一次性 manifest
2. 同步左/右红外各一帧 + RTSI 关节快照 + ChArUco 角点（如果有）
3. **恰好一帧**后立即关闭 session 并标 `completed`——不能追加
4. 同一周期产生 `inference_stationarity.json`，绑定：曝光/推理/占用重建区间的**参考状态 + 后续状态序列 + 判定阈值 + 来源 session manifest 路径与哈希**

---

#### B3. FoundationStereo 推理

**原理**：FoundationStereo 在 rectified-left 像素坐标系出视差，代码把它转回**全分辨率像素视差** + 度量深度 + 有效域掩膜。

**流程**：

1. 推理前先**重核** FoundationStereo 源码 / checkpoint / 模型配置 SHA-256 与运行时一致
2. 推理过程中停稳采样仍在后台跑（覆盖相机曝光、原始数据关闭、推理、占用重建、首次语义读取）
3. 推理结束保存：左右红外原图、全分辨率视差、度量深度、有效域掩膜、rectification 参数、模型出处、逐数组 SHA-256
4. 关闭 session，把推理资产写进同一个周期目录的 `stereo_inference/`
5. **左右一致性**：对每个像素算 `exp(-|d_L - d_R| / threshold)`，这是**确定性一致性分数**（非校准概率）

---

#### B4. 初始叶片前景（mask）

**目的**：在 schema-5 粗模型存在之前，只能用 occupancy-integration-eligible 的 FoundationStereo 深度初始化前景掩膜，**不用参考曲面投影**。

**流程（自动模式）**：

1. 在有效深度域内做**深度连通分量分析**
2. 排除**接触有效域边界**的分量（避免把半个叶片当成前景）
3. 要求最大内部对象相对第二对象有足够**唯一性**
4. **不做腐蚀/开运算**——薄鳍片、自由边、1 像素边界证据不能被拓扑清理删除

**流程（人工模式）**：场景不唯一 → 自动模式 fail-closed；操作员可在 rectified-left 坐标系提供矩形 hint（`--seed-mode component_hint`，只用来选连通分量，不裁剪结果）或硬多边形 ROI（`--seed-mode hard_roi`）。

输出：`outputs/bootstrap_foreground_seed/`，写一次，含 mask、seed、算法策略、输入数组、FoundationStereo 源资产、metadata 的 SHA-256 绑定；读取时重跑算法逐元素核对。

---

#### B5. 至少 3 个几何独立的操作员引导视角（建图）

**目的**：`UNKNOWN` 阻塞运动 → 单视角不能证明遮挡后为自由 → 必须先集齐最少 3 个独立 FREE 投票。

**流程**：

1. 操作员把相机移到下一个**几何独立**位姿（相对每个已接收视角，相机中心 ≥ 20 mm 或光轴转角 ≥ 5°——只换 view_id 不算数）
2. 重复 B1（停稳）→ B2（采集）→ B3（推理）→ B4（前景）
3. 每得一帧就**全量重建三态占用图**（见 B6），不是增量追加
4. 体素需要 3 个独立 FREE 投票才能从 `UNKNOWN` 变成 `FREE`；`OCCUPIED` 一票即占优
5. 机器人在 mapping 中**完全不动**（不能借采集过程移动）
6. 三态投票满足、地图新鲜度满足 → 状态从 `MAPPING` 切到 `MAP_READY`

> **重要**：机器人自遮罩删去的射线及其背后空间**始终是 UNKNOWN**；自遮罩豁免只能通过静态自由 AABB 在特定体素上做局部例外（且 OCCUPIED 仍阻断）。

---

#### B6. 三态占用图全量重建

**原理**：每个新视图都从 `previous_snapshot=None` 起，**按时间顺序积分整段新鲜来源窗口**——不是"在旧地图末尾追加"。

**流程**：

1. 用当前 UTC 时间过滤来源，仅保留 `[now - maximum_map_age_s, now]` 内的视图（默认 5 s，仅为软件起点）
2. 对每个被接收的 FoundationStereo 视图：
   - 左/右一致性证据 + 分数 + 深度范围检查
   - 同步关节 → 标定 FK → `base_T_flange · flange_T_left_ir` 得相机 pose
   - RTSI TCP 校验 FK ≤ 2 mm / 0.3°（不通过则该视图不入图）
   - 渲染 ES68+D435i **最终模型**到左 IR 系，**自遮罩**掉属于机器人的深度像素
   - 自遮罩后的像素按 ray-casting 投票：深度前于几何表面的射线 → FREE 票；与几何表面一致 → OCCUPIED；越界 / 自遮罩 / 无效 → UNKNOWN
3. 写不可变 occupancy 资产（schema-7），含 mapping-context、quality-evidence、所有支撑相机 pose、FK flange、预测/观测 TCP、ES68 关节向量
4. **完整语义读取器**（full semantic reader）重验：raw session、用户立体标定、rectification、FoundationStereo 源码/权重/配置、自遮罩、FK、积分
5. 通过后才发布为新 `OccupancyGeneration`，并原子暴露在 publisher 上
6. 当前 publisher generation 没变就不推进（提交失败时不能"回滚"——已写资产保留作诊断）

---

#### B7. 粗扫科学模型 + 覆盖账本（直到 schema-5）

##### B7.1 双面代理 + 鳍片发现

**原理**：第一个被接受的停稳视图创建**密度均衡的体素云** → 估计两个面内主轴 → 用 `estimated_thickness_m` 把不可见面外推 → 三种独立安全裕度（可见 / 隐藏 / 切向）；代理中心是规划体中心，**不是质心声明**。

##### B7.2 鳍片发现（不是单法向扫过主面）

**原理**：鳍片不是"主面突起的几何"，而是两侧各有一个物理表面 + 根部 + 自由边。

**流程**：

1. 沿代理面内主轴生成 **±15° 成对斜视**候选（每侧）
2. 工作空间 + ES68 端点 IK 筛选
3. 粗扫代际只有**同时满足**：正反面最小视图数、代理块覆盖、每侧至少一对斜视、每侧伸出鳍片的两个物理表面都在融合曲面里取得证据——才能**单向提升为 schema-5**
4. 任一门限未达 → 继续粗扫或显式阻断，**不进入精扫**

##### B7.3 粗曲面重建

**原理**：基于多视图 FoundationStereo 深度 → 标定 `base` 坐标 → 双面独立 voxel fuse → robust point-to-plane 残差精炼 + 机器人 pose 正则化 → 4 边界 B 样条（root / trailing / tip / leading）→ Coons 不规则域 → 鳍片面/根/自由边分区 → 双面 TSDF 融合（带鳍片测厚保护带）→ marching-tetrahedra 出 mesh。

##### B7.4 覆盖账本（粗阶段）

**原理**：基于 schema-5 粗曲面而非代理平面，记录每个 patch 的样本覆盖 / 残差 RMSE / 局部法向一致性 / 曲率 / 显式质量门失败原因；4 边界 + TSDF mesh 完成度单独报告。

---

#### B8. Schema-5 单向提升 + 粗→精切换

**原理**：schema-5 是粗扫模型最严格的"完成态"——一旦发布，就作为精扫分支的**唯一合法固定参考曲面**。

**流程**：

1. 完成 B7 的所有门限检查
2. 写入 `coarse_model_schema_5/`，含：checksummed fused arrays + 重建 mesh + 元数据 + 关联 generation 路径
3. **单向 + 不迁移**：prepared segment、permit、approval、旧地图 publication、精扫覆盖率**全部丢弃**
4. 精扫协调器实例必须**重新发布新 MAP_READY**，才能提出第一段精扫运动
5. 旧资产**不可变** —— 任何在精扫阶段改 schema-5 的企图都会触发哈希失败

---

#### B9. 下一视点选择（coverage-first）

**原理**：`BladeCoverageNextViewSelector` 只看**累积精扫曲面账本**，不看短时安全占用（那条历史只参与否决不安全短段）。

**流程**：

1. 完成度判定：正反两面全部必需主表面 + 4 边界 + 鳍片双面 + 鳍片根部 + 自由边 → 全部通过覆盖、RMSE、法向一致性门限 → `coverage_complete` 事件
2. 否则用 fixed-reference 投影做候选排序：
   - rectified 相机系下做 look-at / projection / visibility / standoff 检查
   - 持久化标定再 compose 到 `base_T_left_ir` 走 IK
   - 评分 = coverage-first → 关节距离仅做末位 tie-break
3. 候选必须已存 `endpoint_feasible` 六轴关节解与对应 `base_T_tcp`（从停稳关节算），并独立用 ES68 FK 回代验证
4. **不可解**（无候选可达 IK/FK/工作空间） → `NextViewUnavailable` → `MOTION_BLOCKED`，**绝不假装完成**
5. 若当前停稳关节到目标最大单关节差值 > `maximum_segment_joint_delta_rad`，按比例截断，只提出一个中间段（`transit_*` view_id，段后强制重新停稳/采集/规划）

---

#### B10. 短段提议（`SegmentProposal`）

每段哈希绑定：

- 当前实测起点关节
- 本段终点关节 + 最终目标关节
- 目标视点 + 段后采集 ID
- 是否到达最终目标
- 当前占用 generation 的完整身份

关节距离**只决定要不要拆段**，**不代替**端点可达性、碰撞检查、覆盖优先级。

---

#### B11. 两项独立连续扫掠证明

##### B11.1 机器人自碰撞 vs 工作空间

**原理**：关节直线段递归二分；每个区间在中点用 hpp-fcl 算分离距离，再用串联关节链 + 各几何到上游关节轴的保守半径 + 区间关节变化，**给出该区间几何最大位移上界**。

**证书条件**：中点分离距离 − 两侧完整运动上界 − 数值容差 **严格为正**。

**失败**：区间继续二分；达最大深度或最小区间仍无法证明 → 返回 `UNKNOWN`，**禁止把若干无碰采样点升级为通过**。

##### B11.2 机器人 vs 三态占用图

**原理**：把扩张包围球覆盖整个区间机器人几何，逐球检查三态体素；`OCCUPIED` 和 `UNKNOWN` 都阻塞。

**绑定**：机器人几何 hash、地图 content hash、语义 attestation、轨迹 hash、证明参数。

**任一输入变化 → 证据失效**。

---

#### B12. 一段一签的人工批准

**原理**：每个短段必须满足当前 preflight fingerprint 完全匹配。

**流程**：

1. 协调器把当前占用 generation 冻结（`OccupancyGenerationPublisher.freeze()`），冻结期间禁止发布新地图
2. 冻结前后 generation 必须一致（防止批准后又被换）
3. 协调器**精确打印**该段的指纹字符串（绑定：实时起点关节、目标关节轨迹、占用身份、几何 hash、证明 hash、ServoJ 参数）
4. 操作员把指纹原样粘贴回来 → `OperatorApproval` 含操作员身份 + 精确确认串
5. 一次性 permit：消费后过期，不可跨段复用，不可在地图更新后复用

---

#### B13. Guarded ServoJ 执行 + 显式 stop + 停稳

##### B13.1 受控恢复

1. 一次性许可消费后才通过**私有 capability** 准备上电/松闸（不是公开接口 `enable()`）
2. 准备动作**不清除 stop latch、不发送轨迹**
3. 恢复后再次核对实时起点、完整轨迹、占用新鲜度、permit 时效、异步 stop 锁
4. 首个 ServoJ 命令前再核一次

##### B13.2 ServoJ 流

1. 每个 ServoJ 写入前在同一个命令 I/O 门内核对 stop 代次与锁状态
2. `ServoJTime == motion_preflight.servoj_dt_s`（强约束）
3. 若 stop 已经在写入队列里，旧代次许可不能再写下一帧

##### B13.3 段后 stop + 停稳采样

1. 执行器对**同一台机械臂**显式 `stop()`，只有 stop 成功才返回
2. 验证实际关节到达本段目标容差
3. 在完整 `settle_time_s` 窗口内对**任意样本对**算最大关节、TCP 平移、TCP 旋转变化
4. 期间本地单调时钟 + host 状态时间 + controller 时间同时覆盖窗口
5. 任一通道 fail → 再 stop 一次 → `ABORTED`，不自动重发本段
6. 通过 → `AWAITING_CAPTURE`，只接受协调器指定的段后采集 ID

---

#### B14. 循环回到 B2（新视点采集）

把 B2–B13 反复执行，直到：

- `coverage_complete` 事件触发 → `COMPLETE`
- 或 selector 抛 `NextViewUnavailable` → `MOTION_BLOCKED`（**绝不等于 COMPLETE**）
- 或任一阶段异常 → `ABORTED`
- 或运行事件持久化失败 → 不可逆 `FAILED`

---

### C. 写一次顶层链与崩溃恢复

#### C1. 顶层写一次链

```
INIT
  → COARSE_CHECKPOINT+（粗扫每个被接受的 checkpoint）
    → PREPARED（schema-5 + reference 绑定，绑定准备时长）
      → FINE_STARTED（精扫首事件，绑定从 MAP_READY 到首事件的总时长）
        → FINE_CHECKPOINT*（精扫每个被接受的 checkpoint）
          → FINE_COMPLETED（最终覆盖代际 + 严格重放的终态重建）
```

每条事件：run ID、连续 sequence、phase、cycle index、event type、UTC 时间、payload、**前序事件 SHA-256、当前事件 SHA-256**。

#### C2. 恢复（`scan run-unknown --resume`）

**唯一可恢复** = 命令里显式命名的实验根；不会扫 `latest`、不会拼接另一个实验。

**重算内容**：

- 事件哈希、序号、前序关系
- 指向的 run/generation/source authority
- FoundationStereo 推理身份、占用 generation ID、schema-5 路径哈希、固定参考哈希、selector 策略哈希
- 完整语义重读（raw session、rectification、FK、FoundationStereo 源码/权重/配置、自遮罩）

**不恢复**：permit、approval、prepared segment、地图新鲜度、控制器权限；恢复后仍要重新确认物理 stop 并建立新安全证据。

#### C3. 完成态

完成态返回**只读 COMPLETE 报告**，不连机械臂或相机。

---

### D. 只读监督（另一终端）

```bash
uv run bbf supervise replay \
  --snapshot data/experiments/unknown_blade_001/live_timeline \
  --follow
```

- 显示 ES68 关节 / FK 链、计划 ServoJ + TCP 轨迹、停站实际样本（不是高频 tracking）、三态占用体素、FoundationStereo 左右红外 / 深度、粗/精点云 + 多视图并集
- 显示并集**不是 TSDF 融合结果**，停站样本**不是高频 ServoJ tracking**——GUI 在资产语义里明确标注
- 机器人三角网格来自**与活动碰撞检查器相同的 manifest 和 STL**，发布前重新计算 model/collision/robot-geometry 哈希
- 显示的每个点云物理源写入磁盘**追加式注册表**（源路径、文件 SHA、点内容哈希、前序链头），重启会重放并验证完整祖先链；源 / STL / manifest / 链尾变化都会阻止继续发布
- **无审批、无 stop、无机械臂接口**；缺关键证据先写 `BLOCKED` 快照，再让运行器失败关闭

---

### E. 完整状态机（正常路径）

```
IDLE
 → BOOTSTRAP_MAP_REQUIRED
   → WAITING_SETTLED → CAPTURING → INFERRING → PUBLISHING_MAP
     → BOOTSTRAP_MAP_REQUIRED | MAP_READY | MOTION_BLOCKED
       → PLANNING → PREFLIGHTING
         → MOTION_BLOCKED | WAITING_APPROVAL | COMPLETE
           → EXECUTING → SETTLING → AWAITING_CAPTURE
             → WAITING_SETTLED ...
```

**主要失败状态**：

- 感知 / 来源验证 / 地图发布异常 → `FAILED`
- 地图非新鲜 MAP_READY / 下一段不可证明安全 → `MOTION_BLOCKED`
- 执行 / stop / 停稳异常 → `ABORTED`
- 操作员主动中止 → `ABORTED`
- 事件持久化失败 → 不可逆 `FAILED`（清除待审批段，本协调器实例不能恢复）
- 覆盖未完成且无候选 → `MOTION_BLOCKED`（≠ COMPLETE）

---

### F. 物理 vs 软件验收边界

文档反复强调："passing software doctor is not hardware acceptance"。下列项目仍需**人在场、低速、可急停**地记录：

1. 最终 ES68+D435i STL 尺度 / 原点 / 轴向 / 装配姿态 / 保守余量
2. 工作空间边界、叶片支架 / 夹具是否纳入占用或静态障碍物
3. 关节短段上限、速度缩放、ServoJ 周期与跟踪误差
4. Dashboard 启动 stop、段边界 stop、6 类实际/目标速度通道的停稳阈值与反馈新鲜度
5. FoundationStereo 最坏推理时间、3 个启动视角完成时间、schema-5 交接时间、地图重放/预检/人工响应时间（确定全部时序预算与 `maximum_map_age_s`）
6. 自遮罩对真实机器人像素的召回率，以及留下的 `UNKNOWN` 壳是否使合法运动无解
7. 连续网格和连续占用证明在已知安全/已知碰撞轨迹上的假阴/假阳
8. 初始掩模及参考引导掩模对主表面、两只鳍片、全部边界的分区质量
9. 双面重建厚度、表面 RMSE、法向、孔洞、覆盖完成判据

> 验收前 `configs/default.yaml` 保持运动、占用、stop-and-capture 关闭，工作空间与短段上限保持未填写。**不得为了让 doctor 变绿而凭经验填写这些物理量。**

---

### G. 完整工作流图

```text
[离线]
   stereo / hand-eye / MDH 标定
        ↓
   静态自由 / 运动包络 / 几何科学 / 运行时四预算 验收
        ↓
   scan doctor --mode unknown（只读）

[运行准备]
   scan run-unknown（output 目录必须新）
        ↓
   B1: Dashboard stop + 停稳采样
        ↓
   B2: 同步左/右红外（一次一帧）
        ↓
   B3: FoundationStereo 推理（重核源码/权重/配置）
        ↓
   B4: 初始前景（自动 / 矩形 hint / 硬多边形 ROI）
        ↓
   B5: ≥3 个几何独立视角 → 三态投票 → MAP_READY
        ↓
   B6: 三态占用图全量重建（sliding source window）
        ↓
   B7: 双面代理 + 鳍片发现 + 粗曲面 + 覆盖账本
        ↓
   B8: 单向提升 schema-5（不迁移 prepared/permit/approval/旧 publication/旧 coverage）
        ↓
   B9: 下一视点选择（coverage-first，关节距离仅 tie-break）
        ↓
   B10: 短段提议（哈希绑定 generation / 关节 / 轨迹 / view id）

[每段运动]
   B11: 两项连续扫掠证明（自碰撞 + vs 占用）
        ↓
   B12: 一段一签（preflight fingerprint 原样粘贴）
        ↓
   B13: Guarded ServoJ → 显式 stop → 停稳采样 → AWAITING_CAPTURE
        ↓
   ↺ 回到 B2（loop until COMPLETE / MOTION_BLOCKED / ABORTED / FAILED）

[旁路监督]
   bbf supervise replay --follow（只读，断证据则 BLOCKED + 失败关闭）

[崩溃恢复]
   scan run-unknown --resume
   └─ 重新算事件链 / 重新做语义重读
   └─ 不恢复 permit / approval / prepared / 新鲜度 / 控制器权限
```

---

### H. 关键设计哲学回顾

| 原则 | 实现 |
|---|---|
| **写一次 + 哈希链** | 顶层链 / 周期目录 / occupancy generation / schema-5 / 精扫代际全部不可变 |
| **物理源身份** | schema-7 用 session manifest + view metadata + 序号 + frame number，同 view_id 不同帧可分别入图 |
| **物理身份 vs 逻辑身份** | 切 view_id 不算独立视角；改 metadata 立刻被检出 |
| **失败 = 阻断 ≠ 静默回滚** | 任何写失败保留证据作诊断，永不删除或改写 |
| **连续 ≠ 密集采样** | 连续扫掠必须给出区间上界证书，否则返回 `UNKNOWN` |
| **`UNKNOWN` 阻塞** | 自遮罩、新视角不足以投票、未观测空间一律 `UNKNOWN` |
| **一段一签** | preflight fingerprint 必须原样粘贴，permit 一次性 |
| **物理验收独立于软件验收** | `scan doctor` 通过 ≠ 硬件放行 |
| **单向阶段切换** | schema-5 不迁移旧 permit；精扫必须重建 MAP_READY；恢复不迁移任何权限 |

---

## 4. 一句话总结

> **BiBladeFusion 是一个"用机器人眼（ES68+D435i）在手、用 FoundationStereo 深度做脑子、用自研三层监督（科学/安全/时序）做刹车"的工业薄壁叶片双面三维重建系统；代码已完成到"软件预验收边界"，所有能力都建立在"写一次 + 哈希链 + 物理源身份 + 连续保守证明 + 时序验收"的硬约束之上，唯一剩下的就是真机在场、可急停、低速地把上述合同逐一兑现成物理证据。**
