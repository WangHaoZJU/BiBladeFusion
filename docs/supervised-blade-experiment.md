# 未知叶片监督式闭环实验

本文档定义 BiBladeFusion 在 ES68 末端安装 D435i 时的代码级实验边界。目标是把未知叶片的安全建图、双面几何采集、鳍片精扫、覆盖反馈和逐段运动放入同一条可审计状态机。本文所述“实现完成”只表示软件合同和回归测试闭合，不表示真机尺寸、GPU吞吐或安全阈值已经验收。

## 1. 唯一运行链

```text
操作员将相机放到已知安全的初始可见位置
  → stop并取得停稳证据
  → 左/右红外采集
  → FoundationStereo视差、置信度与深度
  → 初始叶片前景（自动唯一分量，歧义时人工ROI）
  → 至少三个操作员引导的独立视角建立三态占用图
  → 生成/更新叶片科学模型与覆盖账本
  → coverage-first选择一个下一视点
  → 只截取一个有界短关节段
  → 连续机器人网格扫掠与连续机器人—占用体素扫掠证明
  → 精确到当前预检哈希的人工批准
  → Guarded ServoJ执行、显式stop、停稳
  → 新视点再次采集，循环直至正反面、边界及鳍片必需分区完成
```

原生 RealSense 深度不是这条链的备用后端。相机可以同时输出原生深度用于独立比较，但安全占用和科学重建均固定使用 FoundationStereo，避免同一运行中出现未建模的坐标、噪声和质量策略切换。

## 2. 初始未知叶片前景

在 schema-5 粗模型存在之前，不能使用后期的参考曲面投影掩模。`scan bootstrap-mask` 从一份不可变 FoundationStereo 推理资产生成初始掩模：

```bash
uv run bbf scan bootstrap-mask \
  --stereo outputs/stereo_seed \
  --config configs/local.yaml \
  --output outputs/bootstrap_foreground_seed
```

自动模式在有效深度域内进行深度连续分量分析，排除接触有效域边界的分量，并要求最大内部对象相对于第二对象具有足够唯一性。它不做腐蚀或开运算，以免删除窄鳍片、自由边和一像素边界证据。若场景不唯一，程序失败关闭，操作员可提供整流左图坐标中的提示：

```bash
# 只用矩形选择完整深度分量，不把结果裁到矩形内
uv run bbf scan bootstrap-mask \
  --stereo outputs/stereo_seed \
  --rectangle 260,120,1010,680 \
  --seed-mode component_hint \
  --output outputs/bootstrap_foreground_seed

# 把人工多边形内部的全部有效深度作为明确前景决定
uv run bbf scan bootstrap-mask \
  --stereo outputs/stereo_seed \
  --polygon data/annotations/seed_polygon.json \
  --seed-mode hard_roi \
  --output outputs/bootstrap_foreground_seed
```

多边形文件可以是`[[u0,v0], [u1,v1], ...]`，也可以是包含`vertices_uv`的JSON对象。输出目录写一次且不覆盖；mask、seed、算法策略、输入数组、FoundationStereo源资产和metadata均由SHA-256绑定，读取时重新运行算法并逐元素核对。

`configs/default.yaml`中的`bootstrap_foreground`数值只是软件起点。真机需检查叶片主表面、正反两只鳍片、鳍片根部、自由边、前后缘、叶尖和叶根的precision/recall，不能只看整张掩模的像素占比。

## 3. 为什么启动需要人工引导的三个视角

未知空间按`UNKNOWN`处理，而`UNKNOWN`阻塞运动。单个深度视角不能证明遮挡后方为自由空间，因此代码不会从一个初始帧直接自动绕到叶片背面。启动阶段要求操作员用已经单独确认安全的方式移动到至少三个几何独立的姿态；系统在每个姿态完成stop、停稳、采集和推理。只有满足独立视角数、自由空间投票和地图新鲜度后，状态才从`MAPPING`变为`MAP_READY`。

这个人工启动边界不能通过将未知体素改成自由体素来绕过。机器人自遮罩删除的射线及其背后空间也保持`UNKNOWN`。

地图首次达到`MAP_READY`后，粗扫会话从首个已接受叶片视图自动建立双面代理、代理法向视点和
鳍片发现计划。鳍片发现不是用一个统一法向扫过主面：每一侧都沿代理的两个面内主轴生成
`+15°/-15°`成对斜视候选，并先通过工作空间与ES68端点IK筛选。粗扫代际只有同时满足以下
条件才可单向提升为schema-5：正反面最小视图数、代理块覆盖、每侧至少一组成对斜视，以及
每侧伸出鳍片的两个物理表面均在融合曲面中取得证据。任一门限失败都继续粗扫或显式阻断，
不会用“看起来像鳍片”的单面点集进入精扫。

第二、第三个人工启动视角根据相机光心相对首帧代理中面的位置自动标记正反侧；位于中面或
侧别与几何矛盾时拒绝接收。粗扫到精扫只迁移已经提交且重新验证的感知source window和
schema-5路径，不迁移prepared segment、permit、审批、旧地图publication或精扫覆盖率。新的
精扫协调器必须重新发布新鲜`MAP_READY`地图，才可以提出第一段精扫运动。

## 4. 监督式运行器

`SupervisedExperimentRunner`是`StopScanCoordinator`的持久化组合壳。它提供以下约束：

- `step()`只推进一个非运动状态动作；
- `run_until_attention()`在需要bootstrap视图名、需要批准、阻塞或完成时返回；
- 缺少显式注入的运动适配器时，批准请求仍然不能执行；
- `OperatorApproval`必须携带操作员身份和当前预检要求的精确确认字符串；
- 每次只执行当前一个完整视点运动，之后状态必须进入`AWAITING_CAPTURE`；
- `--resume`只从命令中明确指定实验根的顶层追加链恢复；旧permit、审批、prepared segment、
  地图新鲜度和控制器权限全部丢弃，恢复后仍须确认物理stop并建立新的安全证据；
- 操作员stop请求优先锁存并直接到达同一台机械臂的stop边界，不等待感知文件提交。

生产入口建立运动驱动前先通过Dashboard下发控制器stop，并在完整停稳窗口内同时检查运行态、
机器人模式、安全状态、实际/目标关节速度以及实际/目标TCP线速度和角速度。启动证据必须绑定
当前stop generation；任一通道缺失、过期、超阈值或stop generation不一致均失败关闭。正常视点
运动结束使用`writeIdle`锁存stop generation，并用采样关节/TCP稳定性确认采集边界；此时
`runtime_state=PLAYING`本身不表示仍在运动。bootstrap仍要求Dashboard任务为STOPPED。
段执行预算由独立软件看门狗监视；它使用不与ServoJ写入共用命令锁的Dashboard
`stopProgram`通道。若超时停止调用本身不返回、被SDK拒绝或二次stop失败，运行以
`single_segment_emergency_stop_unconfirmed`失败关闭并保存证据，不会宣称已停稳。该通道是软件
防护，不是硬实时或安全等级急停；真机验收必须验证SDK并发性、最坏返回时间和RTSI最终停稳证据。

FoundationStereo一次逻辑采集允许在失败后重试，但重试不是覆盖：每次尝试写入新的
`attempt_<uuid>`目录，失败或取消目录永久保留，只有原子创建的`committed.json`能够选定一个
成功尝试。提交时验证原始session、view metadata、双目推理、停稳、占用和科学资产；在线占用
写入可复用同一进程内已完整验证且文件身份未变的对象，跨进程/恢复读取仍从磁盘重建语义和
射线结果。占用schema-7用session manifest SHA、
精确view metadata SHA、序号和相机frame number组成物理源身份，因此相同逻辑`view_id`的两次
真实采集可以分别进入地图，同一个物理帧换名后仍会被拒绝重复计票。schema-6仅允许历史回放。

运行事件由`StopScanRunWriter`逐条写入不可覆盖文件，并形成前向SHA-256链。只读端通过`ExperimentStatusSnapshot`和游标式`read_experiment_events()`查看状态；这些类型在数据结构上固定`motion_command_capable=false`，不持有机器人、审批器或执行器。

`LiveSupervisionBridge`把已提交状态原子发布给现有`supervise replay --follow`界面。界面可显示
ES68关节/FK链、计划ServoJ与TCP轨迹、停站实际轨迹样本、三态占用体素、FoundationStereo
左右红外/深度、粗扫或精扫当前点云及多视图显示并集。显示并集不是TSDF融合结果，停站样本
也不是高频ServoJ tracking；二者均在资产语义中明确标注。监督桥没有机械臂、批准、执行或
stop接口，缺少关键证据时先写`BLOCKED`快照，再使运行器失败关闭。

界面中的机器人三角网格来自与活动碰撞检查器相同的manifest和STL，并在发布前重新计算
model/collision/robot-geometry哈希；它不是另一套装饰模型。每个被显示的重建点云物理源还会
写入磁盘追加式注册表，绑定源路径、文件SHA、点内容哈希和前序链头。重启会重放并验证完整
祖先链，源文件、STL、manifest或链尾变化都会阻止继续发布。该注册表只赋予显示可追溯性，
不产生运动权限。

## 5. 连续扫掠不是密集离散采样

生产运动必须同时通过两个相互独立的连续保守证明：

1. ES68、D435i和连接件的实际碰撞网格相对于自身及配置工作空间几何的完整关节段扫掠；
2. 同一机器人几何相对于新鲜三态占用图的完整关节段扫掠，其中`OCCUPIED`和`UNKNOWN`都阻塞。

实现对关节直线段递归二分。每个区间在中点计算FCL距离，并利用串联关节链、各几何到上游关节轴的保守半径及区间关节变化，给出该区间内几何最大位移上界。只有“中点分离距离 − 两侧完整运动上界 − 数值容差”严格为正，区间才取得证书；否则继续二分。达到最大深度或最小区间仍无法证明时返回`UNKNOWN`，不能把若干无碰撞采样点升级成连续通过。

占用证明使用与当前地图完全相同的体素几何、膨胀和未知空间策略，并绑定机器人几何hash、地图content hash、语义attestation、轨迹hash和证明参数。任一输入变化都会使证据失效。

机器人自遮罩会使其自身及背后体素保持`UNKNOWN`。为了避免机器人被自己的未观测体素永久
锁死，代码支持“已验收静态自由AABB”，但它不是一般UNKNOWN豁免：只有整个体素严格落入
验收AABB才可按外部障碍为空处理，`OCCUPIED`在AABB内仍然优先阻断。验收资产写一次并绑定
操作员、工作站、UTC时间、最终机器人几何hash、工作空间和精确区域。区域必须在整次实验中
不可能出现叶片、夹具、支架或其他外部物体；不能把叶片可能占据的包络声明为空。该资产只
解决特定自遮罩语义，不生成运动permit。

物理验收时从`configs/static_free_acceptance.template.json`复制声明，完成所有检查后记录：

```bash
uv run bbf safety record-static-free-acceptance \
  --declaration configs/static_free_acceptance.completed.json \
  --config configs/local.yaml \
  --output data/acceptance/es68_d435i_static_free_001
```

将输出的路径、acceptance ID和完全相同的AABB写入经审查的local配置。`scan doctor`会重新
构造当前碰撞装配并核对几何、工作空间、区域和ID；任何差异都失败关闭。

ServoJ跟踪与停稳误差使用另一份独立验收资产。先复制
`configs/motion_envelope_acceptance.template.json`，用不少于三次代表性试验填写六轴最大跟踪
偏差、六轴stop漂移、反馈/stop确认上限以及实际/目标关节与TCP六类停稳速度阈值，并完成启动
多通道stop、段边界stop、故障stop和急停检查，再记录：

```bash
uv run bbf safety record-motion-envelope-acceptance \
  --declaration configs/motion_envelope_acceptance.completed.json \
  --config configs/local.yaml \
  --output data/acceptance/es68_d435i_motion_envelope_001
```

输出绑定当前机器人几何、碰撞合同和ServoJ控制合同。连续网格证明和占用证明都会按验收后的
关节不确定性扩大包络；任何配置、STL或控制参数变化都要求重新验收。该资产本身仍不授权运动。

## 6. GPU迁移与非运动检查

GPU主机使用项目提供的严格安装脚本：

```bash
./scripts/bootstrap-gpu.sh \
  /absolute/path/to/elite_cs_sdk-1.0.0-cp312-cp312-linux_x86_64.whl \
  configs/local.yaml
```

脚本要求Python 3.12、锁定依赖、Elite SDK本地wheel、FoundationStereo子模块、NVIDIA驱动和CUDA可用；它最后运行`bbf stereo doctor`。这仍不等价于真实推理验收。迁移模型文件后必须对一份已保存双目session运行`stereo infer-session`，核对运行时身份、显存、推理时间、输出分辨率、有效深度比例和左右一致性。

在连接真机前，可运行：

```bash
uv run bbf scan doctor --mode unknown --config configs/local.yaml

# 单独验证已有schema-5之后的精扫分支时使用；未知叶片完整运行不需要预先提供它
uv run bbf scan doctor \
  --mode fine \
  --reference-coarse-model outputs/coarse_model \
  --config configs/local.yaml
```

该命令不打开相机、机械臂或运动驱动。即使所有代码和资产检查通过，最后仍明确保留hardware acceptance警告；检查结果从不生成permit。

通过非运动检查后，完整未知叶片运行入口为：

```bash
uv run bbf scan run-unknown \
  --config configs/local.yaml \
  --output data/experiments/blade-placement-20260831-01-attempt-01 \
  --placement-id blade-placement-20260831-01 \
  --run-id blade-placement-20260831-01-attempt-01 \
  --operator-id vale
```

`placement-id`是夹具和叶片一次不变物理放置的正式身份，并写入顶层INIT哈希链。软件失败但
实物未动时，重试沿用该placement ID，同时使用新的run ID和全新output；夹具或叶片发生任何
物理移动后必须创建新的placement ID。不同placement之间禁止复用占用地图、粗模型、schema-5
参考或精扫覆盖。

输出目录必须不存在；原始双目帧、FoundationStereo结果、占用图、粗扫代际、schema-5、精扫
资产、事件链和监督快照均在其中按写一次语义保存。实验根中的`experiment_handoff`还以
`INIT→COARSE_CHECKPOINT+→PREPARED→FINE_START_CANDIDATE+→FINE_STARTED→FINE_CHECKPOINT*→FINE_COMPLETED`
追加链绑定每次已接受的粗/精扫代际与对应运行事件边界、schema-5及参考模型、精扫候选与首事件，
并以最终覆盖代际和严格重放的终态重建共同封存完成态。链的任一写入、来源验证或重读失败
都不会切换运动权威或报告完成。程序启动后，只有在提示人工初始采集时
按一次`c`才采一组；启用已验收的单视角bootstrap后，后续独立视角由NBV运动自动补齐，而不是
要求操作员手动摆出三个视角。每个完整候选路径都打印当前预检的精确确认串，只有原样粘贴才
消费一次permit。该permit被消费后，私有能力边界才允许上电/松闸准备；准备动作不清除stop
latch、不发送轨迹，随后仍须重新核对关节、地图、新鲜度及绑定的连续证明，才原子恢复ServoJ
控制。已经消费的permit不在恢复后按墙钟再次过期。

另一终端只读观察：

```bash
uv run bbf supervise replay \
  --snapshot data/experiments/unknown_blade_001/live_timeline \
  --follow
```

粗扫到精扫复用同一监督时间线，保留已复制的粗扫点云和停站样本，但清空属于旧协调器的
事件后缀及planned segment；界面始终不能批准、停止或控制机械臂。

进入精扫后，如果安全地图需要新物理源但选择器尚未指定科学`view_id`，控制台会明确要求
按`c`完成一次停站补源。该采集只刷新安全source window，不伪造覆盖观测，也不会自动运动。

崩溃后只能恢复同一个明确命名的实验根：

```bash
uv run bbf scan run-unknown \
  --resume \
  --config configs/local.yaml \
  --output data/experiments/blade-placement-20260831-01-attempt-01 \
  --placement-id blade-placement-20260831-01 \
  --operator-id vale
```

恢复器完整重算事件哈希、序号、前序关系及其指向的run/generation/source authority，不扫描
“latest”目录，也不拼接另一个实验。有限重试的每次尝试使用新的不可变目录，只有原子提交标记
选中一次成功尝试；接受依据绑定原始session manifest、精确view metadata、物理帧身份、停稳和
派生科学/安全资产。完成态恢复只返回只读`COMPLETE`报告，不连接机械臂或相机。

## 7. 科学验收与schema-5时序预算

从模板复制并完成声明。声明不能仅填写汇总指标，还必须指向三份独立的规范JSON证据：
三份证据的字段模板分别是`configs/science_evaluation_report.template.json`、
`configs/science_raw_asset_manifest.template.json`和
`configs/science_independent_review.template.json`；模板仅用于填写，提交前须序列化为下述canonical JSON。

- `geometry_evaluation_report`：评测程序生成，包含生成器名称/版本、实际测量值、样本数、
  距离/入射角包络，并绑定本次runtime contract和原始资产manifest的SHA-256；
- `raw_acceptance_asset_manifest`：按`(role, asset_id)`排序，至少各含一项
  `depth_reference`、`annotation`和`specimen`，每项记录安全相对归档路径、字节数和内容SHA-256；
- `independent_review_report`：复核者必须不同于采集操作员，明确记录`accepted`结论、非空说明、
  完整checklist，并绑定前两份证据及runtime contract的SHA-256。

三份输入禁止重复键、`NaN/Infinity`、重复物理资产或非规范排序，必须使用UTF-8、排序键、
无多余空白且末尾单换行的canonical JSON。记录命令会将三份证据复制到验收目录固定的
`evidence/`子目录，保存字节数和SHA-256；以后每次读取都会重验副本、声明数值/样本数和全部
交叉绑定，外部原路径不参与验收身份，也不能在记录后替换副本。

交叉SHA不能凭空填写。先在最终GPU环境导出实际runtime identity（该命令不连接硬件）：

```bash
uv run bbf safety science-runtime-contract \
  --config configs/local.yaml \
  --output data/acceptance/runtime_contract.json
```

填写raw manifest模板后先做语义校验和规范化；命令拒绝重复键、NaN、缺role、重复资产、
非安全路径及乱序记录，且绝不覆盖已有输出：

```bash
uv run bbf safety canonicalize-science-evidence \
  --kind raw-manifest \
  --input data/acceptance/raw_manifest.completed.json \
  --output data/acceptance/raw_manifest.canonical.json
```

将runtime命令打印的contract SHA和上一步打印的raw-manifest SHA交给机器评测程序；评测程序写完
数值与样本数后，再规范化evaluation并取得其最终SHA：

```bash
uv run bbf safety canonicalize-science-evidence \
  --kind evaluation \
  --input data/acceptance/evaluation.completed.json \
  --output data/acceptance/evaluation.canonical.json
```

独立复核者使用同一contract SHA、raw-manifest SHA和evaluation SHA生成review，最后执行：

```bash
uv run bbf safety canonicalize-science-evidence \
  --kind review \
  --input data/acceptance/review.completed.json \
  --output data/acceptance/review.canonical.json
```

三条命令都打印最终文件SHA和字节数。声明中的`evidence`路径必须指向这三个`.canonical.json`
输出；任何内容修改都必须按`raw manifest -> evaluation -> independent review`顺序重新运行并更新
所有下游SHA。

然后记录不可变验收：

```bash
uv run bbf safety record-science-acceptance \
  --declaration configs/science_acceptance.completed.json \
  --config configs/local.yaml \
  --output data/acceptance/geometry_science_001
```

声明的工作距离和入射角范围必须覆盖当前配置能够产生的完整科学视点包络。验收资产绑定当前
标定、FoundationStereo源码/权重/模型配置以及前景、粗扫、融合、分区、终态重建和选点策略；
同时绑定`pyproject.toml`、`uv.lock`、全部已安装Python distribution的规范名称/版本、OS/内核/
架构/libc、Torch/CUDA/cuDNN/GPU及可读取的NVIDIA driver身份。绝对路径不参与identity，但任何
源码、模型、标定、策略或实际运行环境变化都会使contract失效。把输出路径和acceptance ID成对写入
`science_acceptance.path`与`science_acceptance.acceptance_id`。资产不授权运动，但缺失、范围不足
或contract不一致会使`scan doctor --mode unknown`在打开硬件前阻断。

四项运行时上界不能只靠手写配置获得效力。`configs/runtime_timing_trace.template.json`仅用于
查看schema，不能手工填写后作为证据；一组trial必须各有且仅有一条
`perception_cycle_trace`、`operator_reposition_trace`、`segment_execution_trace`和
`schema5_handoff_trace`。至少完成三组唯一trial，且必须同时包含cold与warm模式。首轮验收不依赖
被它自己阻断的`run-unknown`入口：FoundationStereo感知项从目标GPU对保存session的完整周期取得，
重定位项由现场单调时钟记录相邻停站采集间隔，段执行项来自低速、在场、可急停的受控运动验收，
schema-5项从已保存粗扫代际运行生成/重读/精扫构造窄化试验取得。每份trace的`duration_s`必须由
计时程序写入，不能由操作员编辑汇总值。

在目标GPU主机、目标workcell和同一次Linux启动中先建立密封测量会话；该命令不连接机械臂：

```bash
uv run bbf safety begin-runtime-timing-session \
  --config configs/local.yaml \
  --host-run-id bbf-timing-host-001 \
  --workcell-id es68-d435i-cell-001 \
  --output data/timing/measurement-session-001.json
```

窄化试验程序先调用`write_runtime_timing_measurement_session`生成不可覆盖的canonical session资产，
或直接读取上述CLI生成的同一资产，再调用`biblade_fusion.storage.measure_runtime_timing_trace`，把上述四种真实操作分别作为`operation`
回调，并由`operation_evidence_path`回调返回该次输出资产或运动事件的canonical JSON文件。库会读取
并内嵌该文件，自己计算kind/SHA-256/字节数，不接受调用方直接提交摘要。该API用同一进程的
`monotonic_ns`记录起点、终点和整数纳秒差，验证精确差值后排他写入trace；操作异常、非正间隔、
非canonical操作证据和同名输出都会失败。trace v2同时固化测量实现、当前runtime contract、Linux
boot身份及密封measurement-session；聚合会拒绝跨配置、跨代码/模型环境、跨boot、跨session、
篡改操作证据或复用同一操作证据的trace；最终acceptance的workcell必须与该密封session
中的workcell完全相同。这里刻意不提供“输入一个秒数”的CLI，也不让通用命令替
操作员发起机械臂运动；段执行trace仍只能由现场低速、可急停的受控验收程序包裹真实执行函数产生。
最终acceptance再次绑定当前感知/控制/地图/科学contract，checklist确认目标控制器和workcell。
任何环境变化都必须重新采集，不能仅重新运行聚合命令。

完成全部trace后，按经审查的最坏值和安全余量把四个候选上界写入Git忽略的`configs/local.yaml`。
下面的聚合器会逐个拒绝超过候选上界的trial；随后记录的不可变验收资产会绑定这些精确数值。
只有配置值而没有匹配的资产仍然失败关闭，因此这一步不是用手填数字代替测量。

将十二份或更多trace交给纯聚合命令；该命令不连接相机或机械臂，并拒绝缺role、重复trial/物理
内容、混合host run、非正数、非canonical JSON及超出当前配置上界的结果：

```bash
uv run bbf safety build-runtime-timing-report \
  --config configs/local.yaml \
  --trace data/timing/trial-cold-001/perception.json \
  --trace data/timing/trial-cold-001/reposition.json \
  --trace data/timing/trial-cold-001/segment.json \
  --trace data/timing/trial-cold-001/schema5.json \
  --trace data/timing/trial-warm-001/perception.json \
  --trace data/timing/trial-warm-001/reposition.json \
  --trace data/timing/trial-warm-001/segment.json \
  --trace data/timing/trial-warm-001/schema5.json \
  --trace data/timing/trial-warm-002/perception.json \
  --trace data/timing/trial-warm-002/reposition.json \
  --trace data/timing/trial-warm-002/segment.json \
  --trace data/timing/trial-warm-002/schema5.json \
  --trial-report data/timing/runtime_timing_trials.json \
  --raw-timing-manifest data/timing/runtime_timing_manifest.json
```

聚合输出分别对应`configs/runtime_timing_trial_report.template.json`和
`configs/runtime_timing_raw_manifest.template.json`的固定schema。完成
`configs/runtime_timing_acceptance.template.json`全部检查后记录不可变资产：

```bash
uv run bbf safety record-runtime-timing-acceptance \
  --config configs/local.yaml \
  --declaration configs/runtime_timing_acceptance.completed.json \
  --trial-report data/timing/runtime_timing_trials.json \
  --raw-timing-manifest data/timing/runtime_timing_manifest.json \
  --trace data/timing/trial-cold-001/perception.json \
  --trace data/timing/trial-cold-001/reposition.json \
  --trace data/timing/trial-cold-001/segment.json \
  --trace data/timing/trial-cold-001/schema5.json \
  --trace data/timing/trial-warm-001/perception.json \
  --trace data/timing/trial-warm-001/reposition.json \
  --trace data/timing/trial-warm-001/segment.json \
  --trace data/timing/trial-warm-001/schema5.json \
  --trace data/timing/trial-warm-002/perception.json \
  --trace data/timing/trial-warm-002/reposition.json \
  --trace data/timing/trial-warm-002/segment.json \
  --trace data/timing/trial-warm-002/schema5.json \
  --output data/acceptance/runtime_timing_001
```

把输出path/ID成对写入`stop_and_capture.runtime_timing_acceptance_path`与
`runtime_timing_acceptance_id`。记录命令不信任manifest中的摘要：它重新读取全部`--trace`，逐条核对
role/name/SHA/字节数与聚合报告，并把canonical原始trace复制到验收资产的`evidence/`目录；reader每次
以一次regular-file快照重验文件集合、内容、trace v2的runtime/boot/session/workcell/
operation-evidence身份和报告可重现性，并拒绝核心文件符号链接。资产绑定
四个上界以及当前感知、控制、占用和科学runtime contract；
记录过程是失败关闭且永不覆盖的；若进程在目录发布中途终止，可能保留claim、空目录或
`.incomplete`证据。程序不会自动删除或重用该路径；查明原因后必须选择新的验收输出目录。
doctor在任何硬件连接前重验。运行时感知超时会在地图publish/commit前失败，操作员采集间隔超限
会丢弃旧source window并要求重新集齐连续视角，计划段先检查时长且实际执行超时会stop，schema-5
超时会stop并保持BLOCKED。顶层链在`PREPARED`记录从MAP_READY promotion开始、覆盖schema-5生成
的prepare时长；精扫首事件首先持久化为非权威`FINE_START_CANDIDATE`，预算终检在临时事件文件
`flush/fsync`之后、原子发布`FINE_STARTED`之前完成。只有最新候选可被提交；若候选后崩溃，恢复保持
`PREPARED`、弃用旧fine run，以`resume_fine_start`单独记录新恢复段，并丢弃旧运动权限与地图新鲜度。
精扫StopScan追加与外层`FINE_STARTED`发布强制使用同一canonical-root的进程内`RLock`和
Linux `flock`；预算终检后还会在锁内最后重读唯一bootstrap事件，所以另一writer不能在外层
原子发布前偷渡采集或运动事件，也不依赖墙钟时间戳推测发布先后。
若FoundationStereo调用本身永久不返回，软件只能把它视为liveness故障，无法在同一Python调用栈内
生成超时事件；但进入推理前机械臂已经stop并通过停稳检查，且在推理返回并通过最终预算检查之前既不
publish/commit新地图也不允许运动，因此该挂起不会转化为未受控运动权限。

## 8. 仍需真机验收的项目

代码完成后仍必须由人在场、低速、可急停地记录以下证据：

- 最终ES68、D435i和连接件STL的尺度、原点、轴向、装配姿态和保守余量；
- 工作空间边界、叶片支架/夹具是否纳入占用或静态障碍物；
- 速度缩放、ServoJ周期、内部轨迹步长与跟踪误差；
- Dashboard启动stop、段边界stop及六类实际/目标速度通道的停稳阈值与反馈新鲜度；
- FoundationStereo最坏推理时间、三个启动视角完成时间、schema-5交接时间、地图重放/预检/
  人工响应时间，据此确定全部时序预算；仅在部署明确要求墙钟TTL时另行验收
  `maximum_map_age_s`；
- 自遮罩对真实机器人像素的召回率，以及它留下的`UNKNOWN`壳是否使合法运动无解；
- 连续网格和连续占用证明在已知安全/已知碰撞轨迹上的假阴性与假阳性检查；
- 初始掩模及参考引导掩模对主表面、两只鳍片及全部边界的分区质量；
- 双面重建厚度、表面RMSE、法向、孔洞和覆盖完成判据。

验收前`configs/default.yaml`保持运动、占用和stop-and-capture关闭，工作空间保持未填写。不得为了让doctor变绿而凭经验填写这些物理量。
