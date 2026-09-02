# Phase 0 性能计时与 attempt-09 离线基线（2026-09-01）

## 1. 结论与验收状态

本轮只完成了低扰动计时基础设施、真实 `attempt-09` 的离线 CPU 基线和热点定位；没有实施 occupancy、schema、stationarity、live worker 或运动算法优化，也没有修改配置、安全/科学阈值、占用语义、stationarity schema、motion approval 或机器人执行路径。

最明确的结论是：当前 Python DDA 射线体素遍历是已测离线路径的首要单核热点。单个真实 source 的 `DepthRayIntegrator` warm wall-time p50/p95 为 `17.669/17.734 s`，CPU-time p50/p95 为 `20.104/20.198 s`。源码审计还确认同一 source window 可能在 rebuild、occupancy write 验证和 occupancy readback 验证中被重复回放。

**Phase 0 仍不能标记为完成。** 新 span 已覆盖任务书要求的代码边界，但这些 online span 是在 `attempt-09` 之后加入的；尚无新一轮真机产物可回答真实 CUDA stereo、完整 production semantic readback、coarse generation、live snapshot、fine IK/FK 和 segment preflight 各自的前三大耗时函数。按 `优化.md` §5.6，缺少 timing 不能视为通过，也不能据此批准缓存、并发或语义级重构。

## 2. 输入身份与只读边界

- Git 基线：`f3044e8 fix: harden unknown-blade bootstrap pipeline`
- 输入：`data/experiments/blade-placement-20260901-01-attempt-09`
- 文件总大小：`867,981,165 bytes`，约 `827.77 MiB`
- cycle 数：`3`
- coarse generation 数：`3`
- 诊断 content-tree SHA-256：`0af5558a329d5829f3fdc67a81e98749ed53bb92cf4e02fc1fdfc53ae9caed4c`

content-tree hash 的计算口径是：按相对路径排序，对每个文件依次加入“相对路径、NUL、文件内容”后计算 SHA-256。它用于固定本次 benchmark 输入，不是项目的安全或科学 authority，也不替代 experiment event chain。

输入目录全程只读。所有 benchmark 输出均写到新的 `/tmp` 目录：

- artifacts 最终报告：`/tmp/bbf-attempt09-phase0-artifacts-3x5-final-root/phase0_benchmark.json`，`52,291 bytes`
- ray replay 最终报告：`/tmp/bbf-attempt09-phase0-ray1-3x5-final-root/phase0_benchmark.json`，`66,577 bytes`
- three-source scaling smoke：`/tmp/bbf-attempt09-phase0-ray3-smoke-final-root/phase0_benchmark.json`，`33,989 bytes`

## 3. 可复现命令

### 3.1 artifacts：cold 3 / warm 5

```bash
/usr/bin/env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python \
  -m biblade_fusion.diagnostics.performance_benchmark \
  data/experiments/blade-placement-20260901-01-attempt-09 \
  /tmp/bbf-attempt09-phase0-artifacts-3x5-final-root \
  --suite artifacts \
  --cold-runs 3 \
  --warm-runs 5
```

### 3.2 一个真实 source 的 ray replay：cold 3 / warm 5

```bash
/usr/bin/env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python \
  -m biblade_fusion.diagnostics.performance_benchmark \
  data/experiments/blade-placement-20260901-01-attempt-09 \
  /tmp/bbf-attempt09-phase0-ray1-3x5-final-root \
  --suite ray-replay \
  --cold-runs 3 \
  --warm-runs 5 \
  --ray-source-limit 1
```

### 3.3 三个 source 的 scaling smoke：cold 1 / warm 1

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/python \
  -m biblade_fusion.diagnostics.performance_benchmark \
  data/experiments/blade-placement-20260901-01-attempt-09 \
  /tmp/bbf-attempt09-phase0-ray3-smoke-final-root \
  --suite ray-replay \
  --cold-runs 1 \
  --warm-runs 1 \
  --ray-source-limit 3
```

“cold”表示每次使用新的 spawned Python 进程；没有清空或控制 Linux kernel page cache。“warm”表示在同一 Python 进程内重复。因此本报告不能把 `read_bytes=0` 解释为没有读取数据，也不能把这里的 cold 称为冷磁盘测试。

## 4. 验证范围

离线 benchmark 是诊断工具，不产生 motion-eligible 对象：

- stereo 使用 production stereo artifact reader。
- occupancy 使用 relocation-safe 本地读取：检查文件 checksum、dtype、shape、snapshot identity，并构造 typed per-frame object；不检查复制数据中指向 eiai 主机的外部绝对 source path，也不声称完成 production evidence/hash chain 验证。
- coarse scan、stationarity 和 generation 在 artifacts suite 中只执行内容解析与 hash，不是 production semantic readback。
- ray suite 使用未修改的 `DepthRayIntegrator` 回放选定 source，并逐 source 与持久化 mapping snapshot 精确比较；比较通过仍不等于完整 production mapping-chain authority。
- benchmark 不连接相机、GPU、机器人、executor 或 approval；GPU memory 标记为 `not_applicable_no_gpu_workload`。

## 5. artifacts cold 3 / warm 5 结果

下表为 wall-time p50/p95；括号内为 CPU-time p50/p95。

| Span | Cold | Warm |
|---|---:|---:|
| `benchmark.total` | `5.085/5.130 s` (`5.084/5.129 s`) | `4.998/5.059 s` (`4.997/5.058 s`) |
| `occupancy.relocated_content_readback` | `4.690/4.728 s` (`4.689/4.727 s`) | `4.618/4.662 s` (`4.617/4.661 s`) |
| `stationarity.trace_content_readback` | `0.169/0.172 s` (`0.169/0.172 s`) | `0.170/0.170 s` (`0.170/0.170 s`) |
| `stationarity.authority_content_readback` | `0.168/0.171 s` (`0.168/0.171 s`) | `0.171/0.173 s` (`0.171/0.173 s`) |
| `stereo.artifact_readback` | `0.054/0.054 s` (`0.054/0.054 s`) | `0.032/0.052 s` (`0.032/0.052 s`) |
| `coarse.scan_view_content_readback` | `0.0045/0.0046 s` | `0.0045/0.0049 s` |
| `coarse.generation_content_readback` | `0.00030/0.00036 s` | `0.00030/0.00031 s` |

按 exclusive wall p50 排序，前三项是：

1. relocation-safe occupancy content readback：`4.690 s`
2. stationarity trace content readback：`0.169 s`
3. stationarity authority content readback：`0.168 s`

这组结果只说明离线本地内容读取约需 5 秒，不能解释真机单 cycle 的数百至上千秒，也不能代表 production full-semantic occupancy reader。

## 6. one-source ray replay cold 3 / warm 5 结果

| Span | Cold wall p50/p95 | Cold CPU p50/p95 | Warm wall p50/p95 | Warm CPU p50/p95 |
|---|---:|---:|---:|---:|
| `benchmark.total` | `25.193/25.274 s` | `27.522/27.656 s` | `25.258/25.272 s` | `27.673/27.721 s` |
| `occupancy.depth_ray_integrator` | `17.587/17.664 s` | `19.982/20.127 s` | `17.669/17.734 s` | `20.104/20.198 s` |
| `occupancy.relocated_content_readback` | `4.680/4.771 s` | `4.680/4.771 s` | `4.696/4.708 s` | `4.696/4.708 s` |
| `occupancy.ray_replay_fixture_read` | `2.430/2.492 s` | `2.430/2.492 s` | `2.470/2.473 s` | `2.470/2.472 s` |

每次 replay 都与保存的 snapshot 精确相等。按 exclusive wall p50 排序，前三项是 DDA integrator、relocation-safe occupancy readback、ray fixture read；DDA 占总 wall p50 约 `69.8%`。

## 7. 资源指标

以下均为 p50/p95。Linux `ru_maxrss` 单位是 KiB；warm trial 共用一个进程，因此 warm RSS 是进程生命周期 high-water mark，而不是每个 trial 独立峰值。

| Suite / mode | Process CPU | Peak RSS | `rchar` | `read_bytes` | Minor faults |
|---|---:|---:|---:|---:|---:|
| artifacts cold | `5.084/5.129 s` | `372,988/373,048 KiB` | `698,415,452/698,415,452 B` | `0/0 B` | `233,651/245,536` |
| artifacts warm | `4.997/5.058 s` | `374,368/374,368 KiB` | `698,415,460/698,415,461 B` | `0/0 B` | `80,628/231,207` |
| ray1 cold | `27.522/27.656 s` | `374,612/376,036 KiB` | `965,890,222/965,890,222 B` | `0/0 B` | `305,636/323,913` |
| ray1 warm | `27.674/27.721 s` | `375,624/375,740 KiB` | `965,890,231/965,890,231 B` | `0/0 B` | `109,477/306,564` |

`rchar` 表示 read-like syscall 返回的字节数；`read_bytes=0` 表示本次读取由 kernel cache 满足，不表示未读文件。两套离线 workload 都没有 GPU 工作，因此没有 GPU memory 数值。

## 8. smoke 与 cProfile 证据

在正式 3/5 基线前做过两个诊断 smoke；它们不是验收统计：

- one-source 1 cold / 1 warm：cold DDA `17.579 s wall / 19.909 s CPU`，warm DDA `17.794 s wall / 20.265 s CPU`；snapshot 精确比较通过。
- three-source 1 cold / 1 warm：cold 总计 `65.950 s wall / 72.678 s CPU`，其中三个 `DepthRayIntegrator` 合计 `58.427/65.156 s`；warm 总计 `66.343/73.402 s`，其中 integrator 合计 `58.739/65.799 s`。三个 snapshot 均精确比较通过。该结果只用于确认耗时会随 source 数增长，不能代替 3/5 分位数。

对一个真实 source 做的 `cProfile` 带来明显 profiler overhead：`259,378,608` 次调用、总计约 `51.4 s`。因此绝对时间不与上表混用，但调用计数和热点排序足以定位 Python 单核成本：

| 函数/操作 | 调用或累计证据 |
|---|---:|
| `_ray_voxel_indices` | self `12.376 s`，cumulative `19.568 s` |
| `_index_in_bounds` | `19,432,958` calls，self `4.524 s`，cumulative `15.817 s` |
| bounds generator expression | `65,587,954` calls，约 `7.304 s` |
| `set.add` | `30,554,197` calls，约 `2.877 s` |
| `min` / `max` | `19,097,611` / `19,097,605` calls |
| `list.append` | `25,087,544` calls |
| `DepthRayIntegrator.integrate` | cumulative `45.286 s`（profiler 下） |
| `time.sleep(0)` | `5,240` calls |

这证明已测 one-source offline replay 的首要 CPU 热点是逐像素射线的 Python Amanatides-Woo/DDA 遍历、重复 bounds check 和 set/list 更新。它不能单独证明整个真机流程的首要瓶颈，也不能排除 FoundationStereo、production readback、coarse generation 或 live snapshot 在真机上的占时。

源码审计还发现一个独立的重复工作候选：一个 source window 在 `_rebuild_updates()` 中积分后，`write_occupancy_mapping()` 的预写验证会再 replay，随后 `read_occupancy_mapping()` 的完整 integrity/semantic 验证还会再 replay。第三个 bootstrap 的三个 source 因而可能在一个事务内触发约九次积分。这个判断用于确定下一阶段测量方向；本轮没有删除或绕过任何 replay/readback。

## 9. 已加入的在线 span

计时实现固定最多 `64` 个 span 名，只保留 count、failure count、inclusive/exclusive wall/CPU aggregate 和小型资源快照；不保留逐 ray、逐机器人样本或逐调用 trace。诊断文件明确标记 `diagnostic_only_not_safety_or_science_authority`，使用 no-clobber 原子发布，不进入 safety hash、science hash、approval token 或运动 authority。

已覆盖代码边界如下：

- perception：cycle total、stereo backend、stereo write/readback、current-frame prepare、source-window selection/rebuild、historical/current source integration、occupancy write/readback、fine assets、coarse preflight/scan-view prepare/readback、sampler finish、stationarity trace write/validation/authority write-readback。
- occupancy：`DepthRayIntegrator.integrate` aggregate。
- coarse science：foreground、reconstructed view、reconstructed/scan-view write、generation source/previous/coverage read、coverage update/write、generation write、generation acceptance、proxy build、initialization、initial/discovery candidate filter 和 plan write。
- experiment handoff：coarse checkpoint append、完整 verify。
- live supervision：perception ingest、collision geometry、voxel conversion、asset binding、array/file write、snapshot publication/commit。
- planning：generic `filter_candidate_views`、每候选 reachability aggregate、fine blade FK filter、endpoint FK consistency、live joint preflight、linear joint preflight、stop-scan next-view selection 和 segment prepare。

instrumentation 不会放宽任何 gate、生成授权或把失败改写为完成；普通计时错误和诊断 I/O 错误不会替换主操作的返回值或异常，`KeyboardInterrupt`/`SystemExit` 也不会被可选计时吞掉。诊断文件仍在部分同步调用返回前执行一次小型序列化、文件关闭和 no-clobber 发布（诊断路径不做 durability `fsync`），因此仍存在测量扰动，并可能使外层 perception duration 更长、在预算边界触发更保守的阻塞；不能把“非权威”误解为“零时延”。

## 10. 尚未获得的真机数据

下列边界已有 span，但 `attempt-09` 运行时尚未包含这些 span，所以必须等下一次真机测试产生新诊断后才能验收：

- FoundationStereo CUDA backend、stereo artifact write 与 production readback；
- current-frame occupancy prepare、三个历史/current source replay、occupancy writer 内验证和 production full-semantic readback的独立占比；
- foreground、proxy、reconstructed view、coarse scan view 和 generation coverage update/write/read；
- sampler finish、两份 stationarity 写入与严格 readback；
- experiment checkpoint append/full verify；
- live ingest、collision mesh transform、voxel conversion、assets 与 snapshot commit；
- 真正的 coarse/fine IK 候选数、每候选 reachability、fine FK filter；
- 一个真实短段的 continuous mesh/occupancy segment preflight。

下一次真机结果还必须核对诊断缺失或写失败；缺 timing 时应把性能验收判为未完成，不能自动降级为“通过”。no-clobber 策略在相同 diagnostic path/identity 重试时保留第一份文件，后续写入返回失败而不覆盖旧记录；因此不能把首份 timing 冒充最新重试的测量，发生这种情况同样必须判为该次 measurement acceptance 未通过。

## 11. 后续最小建议（仅提案，未实现）

本轮只提出两个后续方向，不施工：

1. **Phase 4 compiled DDA**：在保持同一离线 oracle 的前提下，把 `_ray_voxel_indices`、bounds check 和 set/list 热循环迁移到经过逐 snapshot 等价验证的 compiled/vectorized 边界。必须对 free/occupied indices、vote counts、source identity、sequence、evidence hash 和 blocking reasons 逐字段比较，任何不一致即停止。
2. **按不可变 source authority 绑定的 per-source contribution cache**：缓存 key 至少包含 source 内容 hash、mapping geometry/context、integration policy和实现版本；只复用已验证的单 source 贡献，并在关键 motion/MAP_READY/schema-5/resume/final 边界保留独立 authoritative replay/readback。禁止按路径、mtime 或对象身份做全局缓存。

在拿到第 10 节的真机 span 之前，不应开始上述重构，也不应先上八叉树。当前数据首先支持优化重复 replay 和 Python DDA，而不是改变 occupancy 表示或安全语义。

## 12. 本轮验证

定向 Ruff：所有本轮性能相关文件通过。

定向 pytest（集合有重叠，不能相加成总数）：

- diagnostics、planning、IK/FK、motion preflight、stop-scan coordinator：`119 passed`
- diagnostics、integrator、FoundationStereo cycle、unknown-blade coarse/runtime、live supervision：`149 passed`
- live/checkpoint/diagnostics 初始组合：`77 passed`

这些测试证明包装层、O(1) aggregate、no-clobber 诊断、异常传播和既有定向行为未回归；它们不替代下一次真机性能与安全验收。
