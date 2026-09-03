# 双面带鳍叶片的覆盖驱动下一视点选择

## 1. 作用与边界

`BladeCoverageNextViewSelector`把固定粗模型、独立精扫覆盖证据和ES68端点可达性连接起来，
每次只输出一个精扫目标。它不负责运动执行，也不把安全占用图当作重建完成度。

系统维护两条语义完全不同的历史：

- 安全占用图只保留仍在新鲜时间窗内的FoundationStereo观测，用于后续短段轨迹碰撞预检；
- 曲面覆盖账本从一个固定coarse schema-5参考开始，按精扫观测永久累积，用于判断真实曲面是否
  已达到覆盖率、RMSE和法向一致性门限。

占用图可以否决一个已经选出的短段，但不能提高曲面覆盖率、改变候选排序或宣布扫描完成。
当一个远目标被拆成多个短段时，`transit_*`停稳采集只重建安全占用，并沿用上一份已经完整
校验的精扫代际；它既不制造空的科学观测，也不重置累计覆盖。到达正式参考候选视点后，本轮
必须同时产生FoundationStereo叶片重建和对应的覆盖后继代际，否则按科学资产缺失失败关闭。

## 2. 固定参考与独立精扫代际

`write_surface_coverage_generation()`生成不可覆盖写入的精扫代际：

```text
schema-5 coarse model
  └─ generation 0：空精扫账本，不导入任何粗扫coverage
       └─ generation 1：generation 0 + 一个FoundationStereo重建视图
            └─ generation 2：generation 1 + 一个FoundationStereo重建视图
                 ...
```

每个后继代际严格绑定上一代、一个`StoredReconstructedBladeView`、参考粗模型metadata哈希、
质量配置和数组manifest。`observation_id`必须同时等于重建视图的`source_view_id`和参考计划中的
候选ID。读取器从第一代开始逐代重放覆盖更新，并独立重算质量；metadata中的`complete`字段
不能单独作为证据。

selector构造时还必须显式固定预期粗模型的规范路径和metadata SHA-256。第一次选择不接受
“任意自洽”的另一叶片代际；同一运行内，每个`transit_*`周期必须精确沿用上一轮已接受的
generation路径和ID，而正式科学观测的后继必须以该generation作为`previous_generation`。
这阻止缓存或路径串线把另一叶片的合法资产误当成本次覆盖，进而错误规划或宣布完成。

只有schema-5粗模型可以作为参考，因为它同时保存：

- `base_T_left_rectified`：投影、视场、入射角和可见性使用的虚拟校正相机姿态；
- `base_T_left_ir`：物理左红外相机和机器人IK使用的姿态；
- `left_rectified_T_left_ir`：两者之间的已标定刚体变换。

读取时逐候选验证：

```text
base_T_left_ir = base_T_left_rectified · left_rectified_T_left_ir
```

旧schema不会被猜测为单位校正后继续参与精扫。

## 3. “完成”的严格定义

默认策略要求正面和反面都包含并完成以下区域：

```text
surface, leading_edge, trailing_edge, root, tip,
fin_face, fin_root, fin_free_edge
```

两侧鳍片还必须满足：

1. 固定粗模型各有且只有其对应的已分割鳍片分量；
2. `two_faces_observed=true`；
3. 每侧`fin_face`候选中，相对鳍片法向轴同时存在正向和负向的物理面；
4. 两组面分区都通过精扫质量门，不能只以“出现了fin_face区域名”代替双面证据。

每个分区通过以下全部门限后才是`complete`：覆盖比例、曲面距离RMSE、局部法向一致性和观测
参考点数。鳍片或遮挡驱动的细分块可能少于全局`minimum_observed_points`，因此绝对点数门限
最多取该块本身的参考点数；覆盖比例门仍按该块全部参考点计算，RMSE和法向门也不放宽。

只有所有策略要求分区均通过时，selector才返回`coverage_complete=true`且不含目标。以下情况
都不能伪装成完成：

- 覆盖不完整但没有可达候选；
- 候选已经拍过一次但质量仍未通过；
- 工作空间未配置；
- IK不可用、无解或返回的关节解不能通过独立FK回代；
- 曲面、鳍片、相机坐标系或代际来源不一致。

前两类进入`NextViewUnavailable`和`MOTION_BLOCKED`；科学资产损坏或不一致进入`FAILED`。

## 4. 候选硬门

selector只从尚未完成且尚未使用其候选ID的分区中取候选，并依次执行：

1. 使用`base_T_left_rectified`复核look-at、入射角、目标距离、投影率和可见率；
2. 使用完整粗曲面点集的保守OBB检查相机与叶片的几何净空；
3. 使用物理`base_T_left_ir`检查工作空间、禁入体和ES68端点IK；
4. 每个周期用当前停稳轨迹的最新关节状态重新构造IK checker，不复用初始姿态seed；
5. 用带关节零位offset的标定ES68 FK回代IK解，并与目标`base_T_tcp`比较平移和旋转残差；
6. 只保留同时通过全部门限且携带六轴关节解的候选。

不同曲面分区即使相机姿态近似相同也不会在精扫selector中被静默去重，因为保留其中一个姿态
并不能自动证明另一个分区获得了有效观测。后续若要合并，必须提供“一次曝光同时覆盖多个分区”
的显式可重算证据。

## 5. 在线科学增益与确定性排序

通过硬门的候选不再仅按覆盖缺口词典序排序，而是计算叶片ROI内的预期科学增益：

```text
measurement_quality = cbrt(visibility * projection * incidence)
coverage_novelty = (1 - coverage) * measurement_quality
quality_recovery = coverage * quality_deficit * measurement_quality
expected_gain = semantic_priority
                * (w_cov * coverage_novelty + w_quality * quality_recovery)
                + unobserved_fin_face_bonus
```

`quality_deficit`取法向不一致和归一化RMSE缺口的较大值。`semantic_priority`由配置中的区域
次序平滑生成，不再作为压倒所有其他证据的硬词典级；尚无观测点的`fin_face`另外获得有限奖励，
使另一鳍片物理面不会被大面积主表面长期淹没。覆盖增加后`coverage_novelty`自然下降，因而相似
视点的重复拍摄会自动降权。全部分量和最终增益都进入下一视点diagnostics，并随协调器事件持久化。

这里复用固定粗曲面生成候选时已经完成的全曲面z-buffer可见率，不对桌面、夹具和背景未知空间
重复进行科学射线计数。安全占用图仍只用于候选选定后的短段路径硬预检，不能通过“背景未知体素
很多”提高科学增益。

selector现在保留完整的科学排序队列，而不只返回第一名。协调器默认最多取前三名，按原顺序逐个
执行完全相同的连续扫掠/占用图硬预检：第一名路径不安全时记录其`view_id`、科学名次和原始
`blocking_reasons`，然后尝试第二名；第一条通过的路径才进入人工确认。占用图不能重算或提升任何
候选的增益，也不能降低安全门限。地图代际错误、证据损坏、时限越界等系统性失败不会触发候选
回退，而是照常立即阻断。若前三名均被路径证明否决，本周期进入`MOTION_BLOCKED`并保留全部拒绝
审计；分段运动中真正通过预检的候选会成为新的稳定目标，不会在下一段偷偷跳回原第一名。

最终按以下顺序确定性排序：

1. 预期科学增益更高者优先；
2. 覆盖新颖性、质量修复量和测量质量依次作为可解释平局项；
3. 几何得分更高者优先；
4. 当前关节到IK解的最大关节变化和总关节变化仅作末级平局判据；
5. 距离偏差及稳定`view_id`完成最终确定排序。

算法版本、完整选择配置、质量配置、视点过滤配置、运动端点门限、运动学配置和
`flange_T_left_ir`共同生成`selection_policy_sha256`。选择结果还绑定精扫generation ID和固定
粗模型metadata SHA-256；短段提议、预检和运行事件继续携带这三个身份。

## 6. 在线科学资产事务与当前边界

`FoundationStereoOccupancyCycleEngine`现已在库级接入明确的叶片科学分支。它不会把全部有效
深度或安全占用体素当成叶片，而是把固定schema-5粗曲面投影到当前`left_rectified`图像，使用
粗模型z-buffer与实测FoundationStereo深度的前、后不对称容差生成前景。输入eligible mask与
占用重建共用质量、量程和机器人自遮罩门；科学mask在此基础上进一步收缩，安全占用则继续保留
所有eligible场景深度。mask算法不做连通域筛选或腐蚀，以免删除薄鳍片、自由边和单像素边界。
目标分区还必须以当前视角达到配置的法向入射余弦，并在该像素赢得完整粗曲面的最近深度
z-buffer；随后才计算目标投影支持和深度匹配率。因而相距仅数毫米的正反表面也不能仅凭落入
同一深度容差而互相冒充，目标实际不可见或被另一面/鳍片遮挡时失败关闭。

协调器为每次采集赋予不可由调用方猜测的`CapturePurpose`：

- `BOOTSTRAP`：新运行在安全地图首次达到`MAP_READY`时创建空generation 0；恢复运行则只携带
  构造时已完整校验的既有generation，不创建或推进代际；
- `TRANSIT`：只刷新安全证据，并精确携带已经接受的精扫generation；
- `SAFETY_REFRESH`：运动受阻后的安全重采；有既有generation时只携带，无既有generation时
  只在本次达到`MAP_READY`后创建空generation 0，任何情况下都不制造候选科学观测；
- `CANDIDATE`：在当前周期内同时产生前景mask、FoundationStereo重建视图和一个覆盖后继。

这些路径先作为候选资产落盘，只有协调器完成独立语义读取、停稳证据检查并接受同一感知事务
后，周期引擎才推进内部source window和accepted coverage路径。失败或取消只丢弃未提交的内存
状态；不可变候选目录可保留用于诊断，但不能被下一周期继承为已接受代际。
在线恢复会递归重放整条覆盖历史，并要求每个非空代际都指向foreground绑定的schema 3重建；
mask会从绑定的stereo、占用integration-valid mask和粗模型重算，点云则由同一深度、mask、内参、
点云配置及`base_T_left_rectified`重新去投影和变换。离线/人工schema 2兼容读取不能作为在线精扫
恢复证据。

公开的`scan run-unknown`已把该分支接入监督式粗扫—精扫composition root，但这仍不是生产
放行。顶层追加链用`FINE_CHECKPOINT`把每个已接受精扫coverage generation绑定到精扫run的
精确事件边界；恢复只采用指定实验链的最后检查点并递归重放全部科学来源，不搜索可变的
“latest”目录。候选失败只触发配置上限内的有限retry，每次retry具有新的不可变采集/推理目录和
原始session、精确view metadata、物理frame、foreground、reconstruction及coverage来源证明；
相同逻辑`view_id`不构成同一物理观测，也不能绕过重复检测。

`blade_foreground.enabled=false`仍是默认值。真实叶片上的mask
容差、鳍片保持率、曲面质量和FoundationStereo运行时间尚未验收。默认
`view_filter.workspace=null`仍会阻断机器人候选；两类连续扫掠证明的真机验收、最终装配尺寸验收
和人工逐段批准也仍是相互独立的生产运动前置条件。
当前owner z-buffer由有限粗曲面点按`projection_radius_px`圆形splat得到，并非三角网格连续光栅化；
因此真机还必须验收各工作距离下的最大投影采样孔隙，孔隙无法被保守覆盖时应失败关闭或改用
三角面z-buffer，不能把离散owner直接表述为连续遮挡证明。
