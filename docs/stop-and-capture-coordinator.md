# FoundationStereo单后端停—算—规划—短移协调器

## 1. 目的与当前边界

本协调器面向机器人眼在手的未知叶片扫描，把感知和运动组织为严格串行的收缩时域闭环：

```text
显式stop并取得有界间隔的停稳采样证据
  → 采集一组双目红外图
  → FoundationStereo推理
  → 从新鲜视图窗口全量重建安全占用图
  → 选择一个下一视点
  → 只预检通向该视点的一小段
  → 操作员批准这一段
  → 冻结本段占用图并执行
  → 显式stop并验证停稳
  → 在新的停稳点重新采集
```

这里的“单后端”是严格约束：周期引擎只接受配置完全一致的具体
`FoundationStereoBackend`，协调器只接受FoundationStereo产生的视差、置信度和深度，
不在运动阶段切换到D435i原生深度。即使原始相机帧同时携带原生深度，协调器也不会把它
用作安全地图或科学重建的替代后端。推理资产还必须通过官方源码、checkpoint和模型配置
SHA-256的语义复验；仅在metadata中写一个后端名称不能通过。

库级协调、资产事务、失败关闭状态机、两项连续扫掠证明及受监督组合入口均已实现。默认配置
仍令`robot.motion_enabled=false`、`stop_and_capture.enabled=false`、
`occupancy.enabled=false`，短段关节上限、工作空间和静态自由区验收也没有填写。因此未经审查
的默认配置必定在非运动doctor阶段失败。本页描述软件合同，不构成真机放行声明。

主要实现位于：

- `biblade_fusion.workflows.stop_scan_coordinator`：状态机、短段提议、地图冻结、逐段批准和
  执行后停稳；
- `biblade_fusion.workflows.foundation_stereo_cycle`：单视图采集、FoundationStereo推理、
  新鲜窗口全量占用重建、可选精扫科学资产事务及完整语义复验；
- `biblade_fusion.workflows.fine_science`：固定粗模型投影mask、精扫重建和覆盖代际准备；
- `biblade_fusion.storage.blade_foreground`：来源绑定、写一次的叶片前景数字资产；
- `biblade_fusion.robotics.stationarity`：停稳窗口和感知计算区间的有界间隔静止采样证据；
- `biblade_fusion.storage.inference_stationarity`：写一次、可重算的感知状态证据；
- `biblade_fusion.storage.stop_scan_run`：写一次API和前向哈希链接的可检出篡改事件资产。

## 2. 为什么不要求FoundationStereo实时运行

FoundationStereo可能需要较长GPU推理时间，并且左右一致性检查需要额外推理。协调器不把
它放入ServoJ控制回路，也不要求它以视频帧率更新地图。机器人只在“有界离散静止采样门已
通过”的状态下进行采集和推理；这句话不表示软件已经获得连续静止证明。一旦开始某个短段，
使用的是已经发布并冻结的不可变占用图。

因此，FoundationStereo延迟直接增加视点之间的等待时间，但不进入ServoJ时序环。每次采集
前协调器先对同一台ES68调用`stop()`并等待停稳；独立只读采样线程在相机曝光前启动，持续
覆盖相机采集、原始数据关闭、FoundationStereo推理、占用全量重建及其首次语义读取，最后
再取一个结束状态。采样结束后，软件才把该轨迹写入stationarity资产并由协调器独立复验、
提交；因此证据不声称覆盖这些后续文件操作。采集bracket状态也按时间合入该轨迹。若任一
已采样状态的关节、TCP平移或TCP旋转超过阈值，控制器状态不是`IDLE`，安全状态不是
`NORMAL/REDUCED`，或状态时间戳/采样间隔不满足新鲜度限制，本次感知结果不能发布为新的
安全地图。

协调器用非重入操作锁串行化公开操作。感知事务未结束时不能启动运动事务，运动事务未结束
时也不能发布新的地图。这是进程内互斥保证，不等同于硬实时控制或动态障碍物检测。

### 2.1 配置同源与防御性副本

“使用同一份配置”不能解释为几个对象碰巧采用相同默认值，也不能依赖调用方一直持有一个
不会被替换的可变引用。推荐的composition root应从同一个已验证`AppSettings`快照构造
协调器、同步采集器、FoundationStereo周期后端和安全工厂；各组件保存所需子配置的防御性
副本。至少以下
安全相关字段必须进入规范化payload和配置身份比较：机器人停稳时间，停稳timeout与poll，
机器人状态最大陈旧时间，采集关节/TCP阈值，FoundationStereo与rectification配置，占用图
几何/年龄/质量策略，以及预检和执行新鲜度策略。

当前构造阶段比较这些防御性副本的Pydantic规范化值，而不是Python对象身份；任一缺字段或
值不同都拒绝组装。软件仅凭等值配置不能证明它们来自同一个YAML文件，因此来源快照一致性
仍由composition root负责，后续应把顶层配置文件哈希加入运行资产。后端资产会绑定可重放的
配置payload，安全工厂的实际运动预检配置也由协调器等值核对。这样，调用方在构造后修改
原始配置对象，不能悄悄改变已组装组件正在使用的策略副本。

启用该协调器还要求`robot.model=es68`、`robot.motion_enabled=true`，并强制
`robot.servoj_time_s == motion_preflight.servoj_dt_s`。FoundationStereo左右一致性阈值必须
非空且不大于占用图契约允许的上限。这些只是组装前置条件；启用YAML不等于运动已经获准，
连续扫掠证明等第7节条件仍会独立阻断。

## 3. 启动阶段：至少三个操作员引导的独立视角

默认占用契约要求：

- `minimum_source_views=3`；
- 一个体素至少获得`minimum_free_observations=3`次独立自由空间观测才可成为`FREE`；
- 新视角相对于每个已接收视角，至少满足相机中心平移或光轴转角中的一个独立性阈值；默认
  阈值分别为20 mm和5°。

所以系统不能从单个初始视角直接宣称拥有可运动地图。`start()`首先进入
`BOOTSTRAP_MAP_REQUIRED`。操作员必须用已经单独确认安全的方式，把机器人置于至少三个
几何独立的停稳姿态并逐次调用采集；此启动阶段不由协调器自动移动机械臂。重复命名、几何
不独立或者已经超出新鲜窗口的图像不能用来凑足数量。

每次bootstrap采集后，若来源不足，地图保持`MAPPING`，协调器继续处于
`BOOTSTRAP_MAP_REQUIRED`。只有完整语义读取器复验得到新鲜`MAP_READY`资产，协调器才进入
`MAP_READY`。

此外，机器人自遮罩留下的体素仍是`UNKNOWN`，而`UNKNOWN`必须阻塞运动。最终真机放行前
仍需解决捕获位姿下机器人自体素/自由壳的严格证明问题，不能直接把自遮罩区域标记为
`FREE`，也不能通过关闭未知空间阻塞绕过该问题。

## 4. 每个视点都是独立的单视图数字资产

`FoundationStereoOccupancyCycleEngine.capture()`为每个停稳视点创建新的周期目录和新的
`SessionWriter`，并在相机曝光前启动状态采样；写入恰好一个同步bundle后立即以
`completed`关闭。协调器在接受结果前再次检查：

1. session manifest必须已经关闭并标记`completed`；
2. session必须恰好包含一个视图；
3. view ID和sequence必须与当前周期一致。

不能在一个开放session中不断追加后续视图。FoundationStereo推理资产会绑定其来源session
manifest、视图metadata和左右红外数组的SHA-256；继续追加会改变manifest，从而使较早的
推理资产失去原始来源一致性。

一个周期当前保存：

```text
cycles/<sequence>_<view>/
  raw/<timestamped-single-view-session>/
  stereo_inference/
  occupancy_mapping/
  inference_stationarity.json
  blade_foreground/       # 仅正式CANDIDATE
  reconstructed_view/     # 仅正式CANDIDATE
  surface_coverage/       # generation 0或正式CANDIDATE后继
```

`inference_stationarity.json`绑定曝光—推理—占用重建采样区间的参考状态、后续状态序列、
判定阈值、来源session manifest路径与哈希，并保存可重算指标和内容哈希。协调器不会只信任
返回对象或文件SHA；它重新读取该文件，复算状态证据，并逐项核对view/sequence、阈值、
manifest、完整状态序列和采集before/selected/after状态，然后才允许发布占用generation。

协调器从状态机而不是视图名称调用方推导typed `CapturePurpose`。操作员引导建图为
`BOOTSTRAP`；中间短段后的`transit_*`为`TRANSIT`；到达正式参考候选为`CANDIDATE`；
`MOTION_BLOCKED`状态下额外采集为`SAFETY_REFRESH`。capture对象和perception结果必须携带同一
purpose，跨边界漂移直接失败。

启用精扫科学分支且显式固定schema-5粗模型后，新运行第一次达到`MAP_READY`的`BOOTSTRAP`
或无既有代际的`SAFETY_REFRESH`周期在本周期创建空generation 0；非`MAP_READY`安全结果不被
科学分支阻塞。恢复运行的`BOOTSTRAP`、`TRANSIT`和已有代际的`SAFETY_REFRESH`不产生mask或
重建，只携带构造时已完整校验的上一份generation；`CANDIDATE`则必须在同一周期内同时生成`blade_foreground`、
`reconstructed_view`和其唯一覆盖后继。前景只接受与固定粗曲面投影深度一致且通过占用质量/
量程/机器人自遮罩门的像素；目标块还必须朝向相机并赢得全表面z-buffer可见性所有权。
自动科学重建写为foreground绑定的schema 3，并完整核对raw/rectified相机链；安全占用仍使用
全部eligible场景深度。schema 3读取还会用绑定深度、mask、内参、点云配置和
`base_T_left_rectified`重新生成像素索引与base系点云；恢复代际的整条非空历史均须满足这一
约束，legacy schema 2不能进入在线精扫连续性。独立
`BladeCoverageNextViewSelector`随后从真实曲面质量、raw/rectified相机链和当前停稳关节产生
候选，并把选择策略、参考粗模型和代际哈希绑定到短段提议。详细契约见
`docs/coverage-next-view-selector.md`。

这条分支仍默认关闭。受监督composition root只会在粗扫完成schema-5并独立读回后创建新的
精扫协调器；粗扫阶段的prepared segment、permit和审批状态不会跨阶段迁移。尚未完成真机
精扫验收。

### 4.1 感知结果的两阶段提交

一个完成推理的返回对象不是已经提交的地图来源。感知事务必须分成两个明确阶段：

1. **候选阶段**：在新的周期目录中关闭单视图raw session，写入stereo、occupancy、
   stationarity以及purpose要求的可选mask/reconstruction/coverage资产。此时它们没有进入
   后续source window，也没有改变当前publisher或accepted coverage；
2. **验证与提交阶段**：从磁盘重新读取候选资产，而不是继续信任内存返回对象。完整语义读取
   必须复验raw来源、FoundationStereo运行时与校准、占用更新链、自遮罩/FK、静止轨迹、配置
   身份、科学资产代际和全部哈希。随后在协调器独占区内再次检查operator stop锁存、当前地图
   generation、capture purpose和预期周期身份均未变化，并在线性化决定后通过publisher事务
   接受对应source window及科学代际，再原子暴露新安全generation；接受异常时先前已接受的
   source window、coverage路径和当前generation均不推进。

状态机的感知/运动公开操作受同一操作锁约束；publisher还用同一把内部锁保护`current`、
`freeze`和事务发布。提交时先在该锁内接受source window，再把对应generation设置为当前值。
因此并发安全消费者在提交结束前只能等待，不能读取或冻结尚未接受的候选generation；接受
失败时当前generation保持不变，无需删除不可变文件来“回滚”。`cancel_pending_capture()`清除
尚未接受的sampler或候选事务状态；已写出的候选目录可作为拒绝证据保留，但不会进入下一周期
source window，也不会更新accepted coverage。`prepare_next_segment()`还要求publisher
generation与当前observation保存的generation ID一致。这里保证的是单进程publisher与感知
引擎之间的事务可见性，不把它表述为跨进程数据库事务。

完整语义重读失败、operator stop请求在提交线性化点前到达、配置身份变化或publisher拒绝
发布时，尚不可见的下一版source window直接作废，候选资产只能保留为诊断/回放资料；它不
得加入当前或下一周期的source window，不得贡献`FREE`票，也不得推进可用于预检的generation。
重试必须从新的周期身份重新开始，不能把失败候选在稍后“补挂”到来源窗口。

提交线性化点只保护很短的“接受或拒绝本次操作”决定；文件读取和可能阻塞的source提交不在
operator-stop锁内执行。`request_stop()`先锁存中止并立即调用同一机械臂实例的`stop()`，不等
publisher或感知提交返回。如果stop在线性化点之后、事务提交返回之前到达，已经通过验证的
不可变资产可以完成接受，但本次运行仍转为`ABORTED`。这一区分的是“运行中止”与“已接受
数字资产回滚”，不能为了迎合运行状态而删除或篡改已经提交的证据。

## 5. 源帧保留与generation替换

每获得一帧新的FoundationStereo资产，周期引擎维护最多
`occupancy.maximum_source_views`帧的滚动来源窗口，并从`previous_snapshot=None`开始按时间
顺序重新积分。与当前帧不满足独立视角门槛的旧帧会被当前帧替换，不能重复贡献`FREE`票；若配置了
`stop_and_capture.maximum_operator_reposition_interval_s`，相邻采集间隔超过该值时才丢弃
此前的前缀。`occupancy.maximum_map_age_s`不再用于删除源帧。

新占用generation只有在来源事务和完整语义验证都成功后才原子发布；失败时已发布generation
保持不变。默认`occupancy.maximum_map_age_s=null`，当前generation不按墙钟过期，只在下一次
成功采集、提交并发布后被替换；部署方显式配置有限TTL时，超时只阻止运动，不删除证据。这样保证：

- 人工引导的三个启动视角不会在推理过程中被墙钟时间删除；
- 每次发布的地图都是具有完整来源链的新一代不可变资产；
- 预检、批准和执行绑定同一个明确generation；失败的新帧不能提前撤销当前generation。

新的占用资产写完后，系统通过完整语义读取器重新验证原始session、用户双目标定、
FoundationStereo来源与配置、手眼标定、ES68 FK、机器人深度渲染、自遮罩和体素积分。只有
得到`full_semantic_verified_for_motion_preflight`的`StoredOccupancyMapping`才可构造
`OccupancyGeneration`。

有限`maximum_map_age_s`是可选部署策略，不是软件默认寿命；一旦配置，必须来自设备实验，且从
generation原子发布时间起算。默认的代际驱动策略仍会在每次授权、执行和freeze边界重验当前
publisher绑定，不能把旧generation跨越一次成功的新发布继续使用。

## 6. 下一视点与短段运动

协调器每次只向`NextViewSelector`请求一个下一目标。具体selector只使用累积精扫曲面账本
判断完成和排序，短时安全占用generation不参与科学得分；占用图只在后续预检中否决不安全
短段。目标必须已经具有端点可达的ES68关节解和对应`base_T_tcp`，并通过独立标定FK回代。
IK seed来自本次停稳感知轨迹的最新关节状态；协调器随后再读取实时关节作为短段真实起点，
不复用初始化姿态。

覆盖尚未完成但没有工作空间/IK/FK均可接受的未使用候选时，selector抛出
`NextViewUnavailable`并进入`MOTION_BLOCKED`，绝不返回空目标冒充扫描完成。只有正反两侧
全部必需主表面、四边界、鳍片双面、鳍片根部和自由边分区均通过覆盖率、RMSE、法向一致性
门限时，才产生带固定参考和策略哈希的`coverage_complete`事件。

若目标与当前状态的最大单关节差值超过实测配置
`maximum_segment_joint_delta_rad`，协调器沿关节直线方向按比例截断，只提出一个中间短段。
中间段用独立的`transit_*`视图ID，并在段后强制重新停稳、重新采集和重新规划。到达最终
目标的短段才应用目标TCP的FK一致性门限；中间段仍必须通过相同的网格和占用安全检查。
`transit_*`周期允许只刷新安全占用并引用上一周期已验证的不可变精扫代际；正式候选ID的
采集则必须在当前周期内同时提交叶片重建和覆盖后继代际。这样短段重规划不会清空或伪增科学
覆盖，而正式精扫漏产资产也不会静默重试。跨周期selector同时钉住粗模型metadata哈希，并
要求transit精确保持generation路径/ID、科学后继精确指向前一代；另一运行的自洽coverage
不能通过这条连续性门。

每个`SegmentProposal`哈希绑定：

- 当前实测起点关节；
- 本段终点关节和最终目标关节；
- 目标视点与段后采集ID；
- 是否到达最终目标；
- 当前占用generation的完整身份。

关节距离只决定是否把目标拆成更短的段，不代替端点可达性、碰撞检查或覆盖优先级。

## 7. 预检、人工批准与地图冻结

每个短段单独调用live joint segment预检。其占用证据绑定：

- map sequence和content hash；
- mapping-context与quality-evidence hash；
- ES68+D435i robot-geometry hash；
- occupancy metadata、语义验证器和attestation hash。

只有预检`ready_for_approval=true`时才创建`GuardedEliteExecutor`并进入
`WAITING_APPROVAL`。每一段都要求操作员输入该段精确preflight fingerprint对应的确认文本；
批准是短时、一次性的，不能跨段复用。

`prepare_next_segment()`只向调用方返回不含执行器、且与内部审批证据深拷贝隔离的摘要，
真实执行器和权威preflight保留在协调器私有状态中，因此调用方修改摘要不能影响后续审批与
执行。协调器、相机
bracket、感知采样、安全工厂和执行器必须绑定同一个机器人对象；安全工厂与协调器也必须
共享同一个占用publisher及第2.1节所述的等值防御性配置副本，构造时Pydantic规范化值不一致
即拒绝。运行资产另行保存可复验payload；当前尚未声称协调器构造本身证明了顶层YAML来源
哈希。

授权和执行期间，`OccupancyGenerationPublisher.freeze()`禁止发布新地图，并验证冻结前后
generation未改变。执行器还会复核当前占用身份、实时起点、精确ServoJ流、运行配置和碰撞
结果。一次性许可消费后，执行器才可通过私有capability清除上一段的stop锁；恢复后会再次
核对实时起点、完整轨迹、占用新鲜度、许可时效和异步停止锁，首个ServoJ命令前再检查一次。
公开`enable()`不能清除该stop锁。地图一旦更新、许可过期或停止请求到达，旧预检和旧批准
都不能继续使用。

异步`stop()`先在独立锁内递增停止代次并锁止，随后才通过短时命令I/O门发送`writeIdle`；它
不等待整段ServoJ运动锁。每个许可绑定当时的停止代次，resume、prepare和每一帧ServoJ写入
都在同一命令门内再次核对代次与锁止状态。若一帧已经进入驱动写调用，它会先于`writeIdle`
完成；`stop()`返回后，旧代次许可不能再向驱动写入任何ServoJ帧。这是进程内命令顺序保证，
不替代控制器急停和硬件安全功能。

当前生产检查器必须同时提供以下两项独立证据：

1. ES68+D435i网格相对于自身和固定工作区的连续扫掠证明；
2. 机器人连续扫掠体相对于三态占用图的连续不相交证明。

网格侧对关节直线段自适应二分，在每个区间使用FCL中点分离距离和串联关节链的保守位移
上界；占用侧以覆盖整个区间机器人几何的扩张包围球检查三态体素。只有严格正的间隙证书
才能通过。达到细分深度、区间宽度或数值界限仍不能证明时返回`UNKNOWN`并进入
`MOTION_BLOCKED`；离散无碰撞采样从不冒充连续证明。两项证据均绑定轨迹、机器人几何、
工作空间/占用generation、策略及证明参数。

机器人自遮罩产生的自身`UNKNOWN`只能在已记录的静态自由区中按例外处理。该例外要求完整
体素落在验收AABB内，并绑定验收ID、机器人几何和工作空间；任何`OCCUPIED`证据仍优先阻断。
静态自由区不得覆盖叶片、夹具、支架或其可能出现的包络，验收记录本身也不产生permit。

## 8. 执行后stop与有界间隔停稳采样证据

每个通过两项连续证明并获得本段精确批准的短段，仍必须按以下顺序收尾：

1. 在一次性许可内受控恢复，并执行精确绑定的ServoJ命令流；
2. 执行器对同一机械臂显式调用`stop()`，只有stop成功才返回；
3. 进入`SETTLING`；
4. 验证实际关节到达本段目标容差；
5. 在完整`settle_time_s`窗口内检查任意样本对之间的最大关节、TCP平移和TCP旋转变化；
6. 进入`AWAITING_CAPTURE`，只接受协调器指定的段后采集ID。

停稳检查不是简单sleep。它同时要求本地单调时钟、host状态时间和控制器时间覆盖配置窗口，
限制相邻状态的最大时间间隔，并计算已采样状态中任意样本对的最大变化。因此它能检出已被
采到的慢漂移和移出—返回行为；等待超时、时钟倒退、反馈冻结、控制器非`IDLE`、状态读取
失败、终点误差或运动阈值超限都会失败关闭。

这仍是离散采样证据，不是数学意义上的连续静止证明：若机械臂在两个状态样本之间运动并在
下一样本前返回，仅靠当前RTSI字段无法排除。证据中的`max_sample_gap_s`只记录并限制最大的
未观测区间，并不会把该区间变成连续轨迹证明；host时间和controller时间都覆盖完整窗口也只
能排除长期旧反馈，不能恢复样本之间的运动。`maximum_robot_state_staleness_s=0.25`是待实测
的保守占位值，不是论文结论。真机放行前必须测得RTSI最坏采样间隔，核验`writeIdle`后的程序
状态，并用控制器实际关节速度/序列信息或“速度上界×最大采样间隔”界证明未观测运动不会
越过允许阈值；否则该证据只能作为感知有效性门，而不能单独作为人员安全功能。

若执行、stop或停稳的任何一步异常，协调器再次尝试stop并进入`ABORTED`；不会自动重发该段。

## 9. 运行事件链

可向协调器注入`StopScanRunWriter`作为`RunEventSink`。每次状态转换追加一个独立JSON事件，
字段包括run ID、连续sequence、phase、cycle index、event type、UTC时间、payload、前一事件
SHA-256以及当前事件SHA-256。

事件先写入同目录临时文件并`fsync`，再用不覆盖目标的硬链接原子发布；写入API拒绝覆盖，
并发写入相同sequence时只能有一个成功。普通文件系统权限仍可能允许外部进程事后修改文件，
所以这里声称的是“可检出篡改”，不是操作系统级不可变。reader不信任`run.json`，而是直接
发现`events/`中的文件，从sequence 0开始重算每个事件哈希、前驱关系、run ID、连续序号、
UTC顺序和有限JSON数据。

`run.json`只保存最新事件路径和摘要，明确标记：

```json
{
  "navigation_only": true,
  "safety_evidence": false
}
```

因此索引损坏或滞后不会改变事件链；事件链本身也只是审计和恢复线索，不是运动批准、占用
语义attestation或连续碰撞证明。

## 10. 状态机与错误语义

正常软件状态序列为：

```text
IDLE
→ BOOTSTRAP_MAP_REQUIRED
→ WAITING_SETTLED → CAPTURING → INFERRING → PUBLISHING_MAP
→ BOOTSTRAP_MAP_REQUIRED | MAP_READY | MOTION_BLOCKED
→ PLANNING → PREFLIGHTING
→ MOTION_BLOCKED | WAITING_APPROVAL | COMPLETE
→ EXECUTING → SETTLING → AWAITING_CAPTURE
→ WAITING_SETTLED ...
```

主要失败状态：

- 感知、来源验证或地图发布异常：`FAILED`；
- 地图非新鲜`MAP_READY`、下一段不可证明安全：`MOTION_BLOCKED`；
- 执行、stop或执行后停稳异常：`ABORTED`；
- 操作员主动中止：`ABORTED`；
- 任一运行事件持久化失败：不可逆`FAILED`，清除待审批段且本协调器实例不得恢复；
- selector以独立重算的精扫曲面质量证明全部必需分区完成：`COMPLETE`；
- 覆盖未完成但没有可用候选：`MOTION_BLOCKED`，不能等同于`COMPLETE`；
- 精扫代际、固定参考、FoundationStereo来源或选择策略语义不一致：`FAILED`。

生产放行前仍要完成：最终ES68+D435i STL尺度/装配与自遮罩真机验收、连续证明的已知安全/
已知碰撞轨迹验证、FoundationStereo/CUDA实测、地图年龄与短段关节上限测量、RTSI采样/
未观测运动边界验收、在线叶片mask/分区/重建质量验证，以及受控硬件验收。完成这些条件前，应把
`MOTION_BLOCKED`视为正确结果，而不是需要绕开的程序错误；硬件急停仍是最终安全边界。
