# BiBladeFusion统一监督台（离线证据回放首版）

## 定位与安全边界

首版监督台只读取不可变、带版本的状态快照，用于离线验收和证据回放。它没有导入
`EliteArm`、`GuardedEliteExecutor`、ServoJ或任何机器人写接口，也不提供批准、执行、暂停、
停止、关节写入或TCP写入控件。即使快照显示外部系统处于`EXECUTING`状态，本界面仍然只是
观察者。

启动方式：

```bash
uv sync --extra supervision-gui
uv run bbf supervise build-replay \
  --occupancy data/occupancy/run_001 \
  --current-view data/reconstruction/view_012 \
  --coarse-model data/coarse-model/run_001 \
  --output data/supervision/run_001/snapshot_0001
uv run bbf supervise replay --snapshot data/supervision/run_001/snapshot_0001
```

`build-replay`是现有数字资产到监督快照的只读桥接。`--occupancy`为必填项；程序沿占用图
中已经校验过的来源链自动读取最后一帧FoundationStereo资产和原始同步session。也可显式传入
`--stereo`，此时它必须正好是占用图最后一帧所绑定的资产。`--current-view`不仅要匹配最后一帧
的view ID、序号和相机帧号，还要匹配session、FoundationStereo资产、关节角、
`base_T_camera`位姿链和手眼标定哈希。`--coarse-model`用于提供融合点云、真实正反面标签、
覆盖率和配准残差；只有其全部源视角均能回溯到本占用图采集链时才标为当前运行已验证，否则
只作为`INDEPENDENT_REFERENCE`显示并增加阻断原因。`--preflight`通过正式schema-5读取器
重新推导，仅显示与同一占用图绑定、且明确已经过期的历史结果。

监管桥使用独立的`ReplayOccupancyMapping`完整性读取类型，只核验回放所需的资产结构、哈希和
内部证据链，并永久标记为`motion_eligible=false`；它不会代替运动预检所用的完整安全读取器，
后者还必须重读语义来源并用当前激活的ES68+D435i模型重新渲染机器人深度。因此，回放能显示
历史占用证据，但无论画面状态如何都不能成为运动放行依据。

生成器先在同级隐藏`.partial`目录中写入所有显示数组，把所引用的小型manifest/metadata
逐字节复制到快照内的`assets/`目录，再逐个校验SHA-256、dtype、shape及快照交叉约束，
最后以一次原子重命名发布最终目录；已有输出不会覆盖。大体量原始采集数据不会重复复制，但
回放所需数组和`AssetRecord`列出的来源描述文件均包含在快照目录中。因此可把每个输出目录
继续作为可搬运的回放数字资产保存。无论输入占用图或历史预检曾经是什么状态，生成结果始终是
`viewer_mode=REPLAY`、`system_state=BLOCKED`、`viewer_motion_command_capable=false`。

桥接会使用占用资产中持久化的`OccupancyConfig.maximum_map_age_s`相对于快照时间重新计算
地图年龄；超过时限就显示为`STALE`，不会把文件中的历史生命周期状态误当作当前可用状态。

占用图只持久化自由/占用整数索引，桥接时按
`origin + (index + 0.5) * voxel_size`转换为体素中心；不会错误地把体素角点作为障碍位置。
未知体素仍由稀疏三态网格隐式定义，首版不会展开为可能极大的点数组，也不会伪造膨胀体素或
探索前沿。

`--snapshot`可以指向：

- 单个`snapshot.json`；
- 包含`snapshot.json`的目录；
- 回放目录，其直接子目录分别包含一个`snapshot.json`。

多帧回放按照快照中的`sequence`排序。播放、暂停、前后切换只改变本地回放位置，不会影响
采集、规划或机器人控制。

如果另一个只读聚合进程持续原子发布子目录，可使用：

```bash
uv run bbf supervise replay \
  --snapshot data/supervision/run_001 \
  --follow
```

`--follow`只轮询目录、验证并追加完整快照；已发布快照若被修改、删除或重排会被拒绝。它没有
相机/机器人连接，也没有确定时延保证，不能称作或代替在线动态避障。

## 两个三维视图

安全场景视图始终可显示由占用证据关节角重算的机器人关节链和历史相机位姿。只有当前最终
ES68+D435i模型能够复现建图时的`robot_geometry_hash`时，才显示世界坐标碰撞网格；否则明确
标为`UNVERIFIED`并保持阻断。只有历史预检通过当前完整读取器重演时，才显示其离散TCP目标
轨迹。实际连续TCP轨迹尚无可靠持久化来源，不会伪造。占用图中的未知体素是隐式集合，不会
展开成可能极大的点数组；视图把未绘制工作空间明确标为`UNKNOWN/BLOCK`，也不会为不存在的
膨胀体素或探索前沿伪造图例。

重建视图分别显示当前深度帧点云和多视角融合点云，并显示正面、反面以及两个鳍片的覆盖率、
注册视角数量和配准残差。安全占用图与科学重建模型保持为两个独立数据产品：前者保守、膨胀
且未知区域阻塞；后者保留高分辨率、不膨胀的曲面几何。

底部标签页显示左右红外图、深度、FoundationStereo左右一致性得分、机器人自遮罩、有效深度率、
一致性得分接受率、LR阈值、FK/TCP残差、数字资产来源、回放构建事件、标定版本和阻断原因。

只有当FoundationStereo推理资产实际携带经过校验的得分数组时，快照才显示该字段。当前得分
定义为`exp(-LR视差误差/LR阈值)`，是确定性的非概率一致性指标，不能解释为标定概率；否则该
字段保持空值。推理延迟、丢帧数以及机器人实际连续TCP轨迹当前没有可靠持久化来源，也明确
保持未知/空值。历史预检必须先通过当前规范读取器的完整重演；监督快照只显示其中已经绑定的
各段`goal_base_T_tcp`平移端点，语义是历史端点折线，不是连续轨迹、实际轨迹或可执行命令。

## 快照契约

契约位于`biblade_fusion.supervision.snapshot`，当前`schema_version`为2。每个快照至少包含：

- 安全状态以及固定为`false`的`viewer_motion_command_capable`；
- ES68状态、关节角、机器人/相机几何和轨迹；
- 三态占用图版本、内容哈希、年龄、分辨率和体素数组；
- 叶片融合模型版本、点云和双面/双鳍片覆盖率；
- 可选传感器图像、占用质量/FK残差、计划进度、数字资产和事件；
- 重建资产的`CURRENT_RUN_VERIFIED`、`INDEPENDENT_REFERENCE`或`UNAVAILABLE`来源状态。

大数组使用相对快照根目录的`.npy`文件，来源描述文件使用相对`assets/`路径；二者都声明
SHA-256，数组还声明精确dtype、shape和语义。读取时会：

1. 禁止绝对路径和目录逃逸；
2. 禁止pickle/object数组；
3. 校验SHA-256、dtype、shape和有限值；
4. 校验每个来源描述文件的SHA-256，并禁止绝对路径和目录逃逸；
5. 校验碰撞网格三角形索引；
6. 强制占用图和重建模型使用机器人`base_frame`；
7. 强制`UNKNOWN`和`STALE`策略均为`BLOCK`；
8. 禁止非`READY`占用图与可批准/执行状态同时出现。

后续在线系统应当由单独的只读状态聚合器原子地产生这些快照。监督台不应直接拼接来自不同
时刻的机器人、相机、地图和计划对象，也不应把“窗口实时刷新”等同于已经实现确定时延的在线
动态避障。
