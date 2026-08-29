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

## 5. 确定性排序

通过硬门的候选按固定词典序排序：

1. 区域优先级：默认鳍片根部、鳍片自由边、前后缘、叶根、叶尖、鳍片面、主表面；
2. 覆盖缺口更大者优先；
3. 法向和RMSE质量缺口更大者优先；
4. 可见率、投影率和几何得分更高者优先；
5. 当前关节到IK解的最大关节变化和总关节变化仅作末级平局判据；
6. 距离偏差及稳定`view_id`完成最终确定排序。

算法版本、完整选择配置、质量配置、视点过滤配置、运动端点门限、运动学配置和
`flange_T_left_ir`共同生成`selection_policy_sha256`。选择结果还绑定精扫generation ID和固定
粗模型metadata SHA-256；短段提议、预检和运行事件继续携带这三个身份。

## 6. 当前尚未接线的环节

精扫覆盖资产和具体selector已经具备确定性单元测试，但
`FoundationStereoOccupancyCycleEngine`目前仍只自动生成raw、stereo、stationarity和安全
occupancy资产。在线生成`reconstructed_view_path`和`coverage_path`还需要一个明确的叶片mask
来源；不能把所有有效深度或安全占用体素直接当成叶片，否则桌面、夹具或背景会污染曲面质量。

在该科学分割/重建事务接入前，selector会因缺少精扫覆盖资产而失败关闭。默认
`view_filter.workspace=null`也会使所有候选停留在几何检查阶段。真实运动还独立受连续
扫掠网格证明、机器人—占用图扫掠证明、真机尺寸验收和人工逐段批准约束；本模块不解除这些
边界。
