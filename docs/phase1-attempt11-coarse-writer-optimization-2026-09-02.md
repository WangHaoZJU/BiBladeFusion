# Phase 1：attempt-11 粗扫视图 writer 严格读回去重（2026-09-02）

> 本文保留最先完成的 writer 单边界证据。随后对真实三帧累计增长的 generation、checkpoint、
> live 和 schema-5 边界进行了扩展分析与修复；当前完整状态见
> `docs/phase1-three-frame-replay-dedup-2026-09-02.md`。

## 状态

本阶段已完成本地实现、语义测试和安全复审；真机采集主机上的 cold 3 / warm 5
前后性能验收尚未执行。2026-09-02 实施期间 eiai 的机械臂和深度相机正在被其他任务
占用，因此没有在 eiai 启动 benchmark、扫描、相机或机器人进程，也没有推送或部署本阶段
改动。

本阶段只优化 `write_coarse_scan_view` 的单个事务边界。配置、安全门槛、科学阈值、
FoundationStereo、DDA 算法、占用语义、IK、workspace、stationarity、运动批准和公共严格
reader 均未改变。

## 输入证据

基线代码提交：`838d6bb2725239ecc2c6f4dbb3eae016dbdb70c1`。

attempt-11 的两个诊断文件属于同一 `operator_bootstrap_000`、sequence `0`、frame
`182`、front/operator-seed 事务：

- `performance_timing.json`：
  `c2c193d41fbaa874576ab9bfad1a0633a07468f74741aa56d7ff1381abd93e78`
- `coarse_generation_timing.json`：
  `fda866a10fed3cca58a452252b13a2b741d801dddd940736ecf188ff2bb2d291`

`performance_timing.json` 状态为 `completed`；`coarse_generation_timing.json` 状态为
`failed/UnknownBladeCoarseError`，对应随后没有可达 fin-discovery pair 的既有安全阻断。
该失败不否定其在阻断前已完成并记录的 DDA timing，但它也不构成一次完整粗扫成功验收。

数据核对结果：

| 项目 | attempt-11 实测 |
|---|---:|
| 第一帧 perception cycle wall | 537.654 s |
| operator foreground preflight exclusive wall | 226.127 s |
| FoundationStereo backend wall | 3.344 s |
| cycle 内 DDA | 7 次 / 220.510 s |
| cycle 内单次 DDA 平均 wall | 31.501 s |
| coarse generation 额外 DDA | 1 次 / 31.380 s |

`foreground_preflight` 的 226.127 秒包含操作者绘制并保存 hard ROI 的等待，不是算法热点。
扣除这段等待后，自动处理约 311.527 秒；FoundationStereo 网络前向不是主要延迟。

源码追踪确认，第一帧 cycle 内 7 次 DDA 中有 3 次来自同一个
`write_coarse_scan_view`：foreground replay、frame identity 验证和最终 integration mask
读取分别触发完整 `read_occupancy_mapping`。这三个读取针对同一个 append-only occupancy
authority。

## 最终实现：3 次降为 2 次

最初评估过把 writer 内三次完整读取降为一次，但发布前若只重验 metadata 和最终 mask，
不能覆盖非最终 occupancy arrays/snapshots、stereo/session、hand-eye 和 active robot
geometry 的事务内变化。该方案没有被接受。

最终采用保守的 `3 -> 2`：

1. 事务开始时调用现有 `read_coarse_integration_source()`。它内部仍调用 production
   `read_occupancy_mapping()`，完整验证 occupancy evidence chain、来源和 active robot
   geometry。
2. 将该冻结的 integration mask 和 frame identity 仅在当前 writer 调用中复用于 foreground
   replay 与 reconstructed-view identity 比较；没有全局缓存，也不跨进程、resume 或事务。
3. 临时 coarse-view 已写好、原子 rename 之前，再次调用同一个 production full reader。
   第二次结果必须与第一次的 view id、sequence、frame、occupancy content hash 和 mask
   完全一致。
4. 第一次读取后绑定的 occupancy `metadata.json` directory record 在 rename 前再次通过
   `_resolve_directory_record` 校验。任一变化都会抛错、删除 `.partial`，不发布产物。

第二次 full reader 会重新检查所有 occupancy arrays/snapshots、stereo/session、hand-eye、
robot source 和 active robot rerender，因此保留了旧实现的完整 authority closure 检查。
它比旧 writer 的最后一次完整读取更靠近 rename。公共 `read_coarse_scan_view`、coordinator、
checkpoint、generation、resume 和 handoff 的独立严格读取均未改变。

本阶段只移除一次重复 DDA。按 attempt-11 单次均值估计，writer 预计节省约 31.5 秒，
cycle 内 DDA 预计由 7 次降为 6 次。该数字是基于实测热点的工程估计，不是真机前后验收
结果。

## 严格 benchmark

新增 `scripts/benchmark_attempt11_coarse_view_write.py`，其规则如下：

- 输入 experiment tree 在运行前后做完整 path/size/SHA fingerprint，且不得写入输入目录；
- 输出目录必须不存在并位于 experiment tree 之外；
- cold 默认 3 次，每次使用新的 non-daemon spawned Python process；
- warm 默认 5 次，在同一 Python process 中复用严格加载的 fixture；
- 只计时 production `write_coarse_scan_view`；fixture load 和严格 post-write reader 位于计时
  区间外；
- 每次输出都由 production `read_coarse_scan_view` 重放，并与 immutable attempt-11 oracle
  逐字段、逐数组比较；只忽略 `created_at_utc`；
- `--expected-dda-count` 是强制参数，每个 cold/warm trial 都必须精确等于 oracle；旧版为
  `3`，本阶段为 `2`；
- 报告记录 wall、CPU、RSS、I/O、产物大小、DDA spans 和 normalized semantic SHA。
- 报告在长测试前后分别绑定 hostname、Git HEAD、相关 `git status`、production
  `coarse_scan.py` 与 benchmark 脚本的绝对路径、size 和 SHA；两份 provenance 不完全
  相同则整次 benchmark 失败。

production DDA 可在内部创建 `ProcessPoolExecutor`。cold worker 因此使用 non-daemon
`ProcessPoolExecutor`，并有真实双层 spawn smoke test。CPU/RSS/I/O 计数来自
`RUSAGE_SELF` 和 `/proc/self/io`，不包含 production 子进程；DDA 的权威性能比较应使用
wall time。

设备空闲后，在仍处于基线提交的 eiai 上先运行：

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/python -B \
  scripts/benchmark_attempt11_coarse_view_write.py \
  data/experiments/blade-placement-20260901-01-attempt-11 \
  /tmp/bbf-attempt11-coarse-writer-baseline-838d6bb-v2 \
  --expected-dda-count 3
```

部署本阶段代码后使用全新的输出目录运行：

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/python -B \
  scripts/benchmark_attempt11_coarse_view_write.py \
  data/experiments/blade-placement-20260901-01-attempt-11 \
  /tmp/bbf-attempt11-coarse-writer-after-phase1-v1 \
  --expected-dda-count 2
```

不得复用先前失败后已存在的 `/tmp/bbf-attempt11-coarse-writer-baseline-838d6bb`。两份
报告必须同时满足：cold `3`、warm `5` 全部完成；DDA count 精确为 `3/2`；normalized
semantic SHA、输出文件数和总大小一致；输入 fingerprint 前后一致。随后再比较 target span
的 cold/warm p50、p95 和 wall-time 降幅。

## 本地验证

- 定向 Ruff：通过。
- writer、粗扫 workflow、FoundationStereo cycle、benchmark：`74 passed`。
- 最终全量 unit suite：`1090 passed, 2 skipped`；跳过项是本机缺少可选 Open3D 的既有
  renderer 测试。
- `git diff --check`：通过。
- 独立安全复审确认：发布前第二次 production full reader 关闭了最初发现的 authority
  closure/TOCTOU 缺口。

## 未完成与下一阶段边界

- 本阶段不能宣称已获得真机 cold/warm 加速数据；eiai 空闲后必须完成上述前后 benchmark。
- 当前仅从约 13 次首帧 DDA 中移除 1 次；coordinator、publisher、commit、generation、
  resume 等边界仍保留各自严格 reader。
- 若要进一步做到 writer `2 -> 1`，应在 occupancy storage 层设计可复用的通用 typed
  authority-closure token，而不是在 coarse writer 内复制 occupancy/stereo/session/robot
  manifest 解析器。该工作必须作为新的单一阶段、独立 benchmark 和独立安全审查执行。
