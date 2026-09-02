# Phase 1：三帧粗扫累计 replay 去重与 GPU 后端边界（2026-09-02）

## 结论

本轮问题不是“只有首帧慢”，而是三个 bootstrap 视角均慢，并且后一个视角会重复验证
前面已经接受的全部 source window。真实 attempt-09 的三帧耗时为：

| 视角 | perception cycle | map state → generation | generation → checkpoint | checkpoint event → live snapshot |
|---|---:|---:|---:|---:|
| 1 | 531.567 s | 145.854 s | 48.912 s | 99.476 s |
| 2 | 991.799 s | 500.410 s | 200.373 s | 302.835 s |
| 3 | 1406.192 s | 1038.734 s | 497.031 s | 642.958 s |

同三帧从 `foundation_stereo_started` 到 stereo metadata 创建的时间间隔约为
`3.351 / 3.044 / 3.044 s`，没有随视角增长；attempt-09 没有单独记录可称为 backend
专属耗时的 online span。最后一列还包含旧版 checkpoint 发布后的完整链复验，不能解释为
纯 live 耗时。
增长主体是 Python DDA 在 writer、generation、checkpoint、live 和粗精交接边界被重复
调用，而不是网络前向。

本轮没有修改 DDA 算法、UNKNOWN/FREE/OCCUPIED 语义、配置、阈值、workspace、IK、
stationarity、运动批准或 FoundationStereo。所做的是在同一事务内传递严格 reader 已产生的
typed readback，并在发布前后重验其不可变 authority；resume、schema-5 handoff 和最终完成
仍执行独立严格重放。

## 三帧增长模型

设第 `N` 帧 occupancy 含 `N` 个 source，完整读取第 `k` 个 coarse view 需要 `k` 次
source integration；`T(N)=N(N+1)/2`，`S(N)=N(N+1)(N+2)/6`。

以下 source-integration 调用模型来自生产调用链复算；attempt-09 的 `1/2/3` immutable
拓扑和耗时增长与它一致，但该 attempt 没有 DDA 在线 count span，不能称为实测调用次数：

- generation：`2N + 3T(N-1) + T(N)`，前三帧为 `3 / 10 / 21`；
- checkpoint：`2S(N)`，前三帧为 `2 / 8 / 20`；
- live：`N`，前三帧为 `1 / 2 / 3`；
- perception/coarse-view 主路径约为 `12N`；
- 第三帧 MAP_READY 后的 transition evaluation 保留两次独立 generation read，即
  `2T(3)=12`；当前配置至少需要 6 个粗扫视角，因此三视角时返回 COLLECTING，并未写
  schema-5 generation。

修改后结构目标：

- perception/coarse-view：`11N`；
- generation：`N + T(N-1) = T(N)`，即 `1 / 3 / 6`；
- checkpoint：仅验证新 checkpoint 的 current generation，`T(N)`，即 `1 / 3 / 6`；
- live：复用同事务 read-only token，额外 DDA 为 `0 / 0 / 0`；
- MAP_READY transition evaluation：第三帧仍为 `12`，未优化、未绕过。

| 帧 | 修改前逻辑 integration | 修改后结构目标 | 消除 |
|---|---:|---:|---:|
| 1 | 18 | 13 | 5 |
| 2 | 44 | 28 | 16 |
| 3 | 92 | 57 | 35 |
| 合计 | 154 | 98 | 56 |

这里的单位是调用结构，不是秒数，不能当作真机加速结论。第三帧仍比第一帧重，因为
occupancy window 本身从 1 增长到 3，并且 MAP_READY transition 必须独立读取 generation；
本轮修复的是
不必要的超线性重复，而不是取消必要验证。

上表统计边界截至三次 capture callback 内的 transition evaluation，不包括随后
`advance_to_attention/select_next` 的额外 `T(3)=6`，也不包括未来某一帧真正满足 gate 后
schema-5 public writer 的额外 `2T(N)`，更不包括完整 fine handoff 的全链读取。因此 `57`
不是“从第三帧一直到精扫启动”的总数。

## 实现记录

### 1. coarse-view writer

`write_coarse_scan_view` 的同一 occupancy authority 从三次完整读取降为两次。第一次严格
读取产生 integration mask 和 frame identity；第二次位于原子发布前，重新验证完整
occupancy/stereo/session/hand-eye/robot closure。旧公共 reader 未改变。

### 2. generation transaction

`CoarseScienceSession` 对当前 view 严格读取一次，对 predecessor generation 严格读取一次；
coverage 更新和私有 writer 复用这两个 typed 对象。`StoredCoarseScanView` 和
`StoredCoarseScanGeneration` 现在保存其实际解析的 authority bytes 的 SHA-256 与大小；私有
writer 使用这些 read-time 记录，而不是在稍后重新抓取一个可能已变化的新记录。发布前再次
校验 view、reconstructed、stereo、occupancy、planning、coverage 和 predecessor authority。

公共 `write_coarse_scan_generation` 仍自行执行严格读取；事务复用入口保持私有，不能由外部
调用者提供任意 dict 绕过 reader。

### 3. checkpoint 增量验证

writer 在 create/resume 时已经完整验证 prefix。追加新的 coarse checkpoint 时：

1. 重新严格解析已有小型 event JSON，验证 schema、sequence、canonical event hash、
   predecessor 和 experiment ID；
2. 持有 coarse run 的跨进程 publication authority lock；
3. 仅对新 checkpoint 绑定的 current generation 执行一次严格 read；
4. 临时 event fsync 后、hard-link 发布前完成上述增量语义验证；
5. hard-link 后再次严格解析 event prefix，并复核 current generation authority。

因此 checkpoint 从 `2S(N)` 降为 `T(N)`。旧 checkpoint 的完整 source replay 不在每次 append
中重复，但在 resume、schema-5 handoff 和 final completion 仍由
`read_unknown_blade_experiment` 完整执行。任何发布前失败都不产生新 event；若极窄的
post-link authority 复核失败，event 已是可审计证据但调用立即失败、writer 不前进，运行
必须 BLOCKED，不能继续下一视角。

### 4. read-only live reuse

coarse acceptance 将同一次严格 view read 绑定为一次性 `_CoarseScanViewReadback`：包含 coarse
metadata 的 read-time SHA/size，以及 reconstructed/stereo/occupancy directory authority。
checkpoint 成功后，adapter 只允许把它传递一次给 `LiveSupervisionBridge`。live 在使用前重验
路径、SHA、大小和与当前 perception result 的 stereo/occupancy identity，然后复制 points 为
不可写数组。token 不持久化、不跨进程、不跨 resume、不参与 coverage、next-view、IK、碰撞或
运动批准；bridge 的 `motion_command_capable` 仍为 `False`。缺失或变更时 fail closed，并回退
不到宽松结果。

### 5. MAP_READY 与 schema-5 独立边界

曾评估过让 `finalize_coarse_generation` 复用前一个内存 generation。安全复审发现仅按 root
匹配不足以覆盖粗精交接 authority closure，因此该方案已撤回。当前仍独立调用 production
`read_coarse_scan_generation`；不能把上表的 12 个第三帧单位当作可随意删除的重复工作。

## GPU 射线投射可行性

GPU 加速技术上可行，并且数据规模适合：attempt-11 单帧的
`occupancy_mapping/metadata.json`（SHA-256
`bf4e746922f73dcfa0f3096069f2a2fa1f4f59dfe0300c524815ce1d7bf7ed75`）绑定的
integration-valid mask（SHA-256
`f405d623799f6cc1625f07664da4a8b7540c7faa3927dbebb3c8724f41ddab17`）产生
`195,484` 条有效采样 ray、`19,310,541` 个 bounds-check 前 DDA voxel-index step
（约 `98.783` step/ray）；后者不等于实际 in-grid bitmap 写次数。网格为
`87×78×92 = 624,312` 个 voxel，一张 byte/bool dense bitmap 约 0.595 MiB。热点位于逐 ray
Python 循环和 Amanatides-Woo DDA，而不是 NumPy 已向量化的反投影和坐标变换。

不能直接把现有 `set/dict` 更新改成无序 CUDA atomic add：当前语义要求同一 source 对一个
voxel 最多投一票、occupied-wins、固定 source 顺序，并且 CPU/GPU 在 grid 面/边/角上的
float64 tie 处理必须逐 voxel 相等。

建议下一阶段保持 `DepthRayIntegrator.integrate()` 和 CPU merge 不变：

1. CPU 按现算法预计算每条 ray 的 `current/target/step/t_max/t_delta/hit/max_steps`；
2. GPU 只执行 DDA 步进，以幂等 OR 写 per-source free/occupied bitmap；
3. 回 CPU 后按固定 C-order linear voxel ID 排序；
4. 使用现有 occupied-wins 和每来源一票逻辑合并；
5. CUDA unavailable、OOM、kernel error、未到 target 或任一 voxel 不同均 fail closed，不允许
   静默 CPU fallback 或近似结果。

第一版应是 PyTorch CUDA shadow backend，因为项目已有 PyTorch/CUDA，不增加 CuPy/NVRTC
部署面；CPU 结果仍为 authority。长期生产版更适合预编译的 PyTorch CUDA custom op，而不是
运行时 JIT。GPU backend 不在本轮生产代码中启用，因为本机和被占用的 eiai 设备尚未完成
逐 voxel、cold/warm、RTSI gap 和故障注入验收。

GPU 验收必须固定本轮去重后的调用 schedule，只比较 backend；记录 CUDA event device time、
端到端 wall、kernel launch、H2D/D2H、ray/voxel visits、显存、GPU utilization/power 和 CPU。
显存占用不等于 GPU 正在计算；请求 GPU 后 kernel count 或 device time 为 0 应直接失败。

## 离线结构报告

新增命令只读取 immutable JSON authority，不连接机器人、相机或 CUDA：

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/python -B \
  scripts/report_three_frame_replay_schedule.py \
  data/experiments/blade-placement-20260901-01-attempt-09 \
  --output /tmp/bbf-attempt09-three-frame-schedule.json
```

本地 attempt-09 报告确认 occupancy source、generation views 和 checkpoint 均为 `1/2/3`，
并把调用链模型应用为 `18/44/92 → 13/28/57`。该脚本验证 immutable 拓扑并套用经源码
复算的公式，不会动态观测 production reader 调用，更不执行 DDA；真实收益仍需 eiai 在设备
空闲后完成三帧 attended run，并按边界检查 online timing spans。

## 验证与未完成项

- 定向单元测试覆盖 writer readback、generation 1/2/3 复用、checkpoint 三次 prefix 不重放、
  event canonical hash、post-link authority、live 一次性传递和结构报告公式。
- repository-wide Ruff 与 `git diff --check` 通过；全量测试为 `1143 passed, 2 skipped`，
  两个 skip 均是本机未安装可选 Open3D 的既有 renderer 测试。
- 本轮未连接 eiai、未启动相机或机器人、未修改任何 experiment asset。
- 真机验收必须执行完整三帧，而不是只测第一帧；比较 perception、generation、checkpoint、
  live 和 schema-5 exclusive spans，并保留 mask、occupancy snapshot、coverage、generation 和
  event chain 的精确等价证明。
- GPU 目前是已完成代码级设计的下一阶段候选，不是已经启用的生产功能。
