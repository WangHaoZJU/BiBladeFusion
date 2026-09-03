# eiai 明日修改清单：机械臂主动叶片测量的五项修复

日期：2026-09-03 计划
适用项目：BiBladeFusion
目标：保留实验阶段需要的必要安全边界，去掉重复或错误的限制，使流程能够从一个人工初始视角开始，自动搜索并采集薄壁双面鳍片叶片。

## 0. 今天确定的总体边界

本系统必须始终区分三类几何：

1. **机械臂自身几何**：ES68、D435i及安装件的URDF原始碰撞STL。
2. **环境安全几何**：叶片、夹具以及相机实际看见的其他外部物体，由全场景占用图表示。
3. **叶片科学几何**：只用于叶片重建、覆盖率和NBV增益计算的叶片ROI/点云。

三个边界不能混用：

- 机械臂STL不能被一个大包围球替代。
- 占用图不能被叶片ROI裁剪。
- 科学重建不能直接使用包括桌面、夹具在内的全场景占用点。

本次实验约定的占用工作空间是：

```yaml
occupancy:
  workspace_bounds_min_m: [-1.00, -0.55, 0.00]
  workspace_bounds_max_m: [ 1.00,  0.55, 1.10]
```

机械臂base的`z=0`略高于桌面，因此本轮实验把`z<0`的桌面排除在占用工作空间之外，不再为桌面增加第二套碰撞模型。

---

## 1. 清除桌面、叶片和夹具的重复碰撞建模

### 1.1 为什么修改

机械臂自碰撞和机械臂对环境的碰撞是两个问题：

- 自碰撞：URDF原始STL之间的Pinocchio/HPP-FCL检测。
- 环境碰撞：同一套机械臂原始STL与占用体素盒之间的HPP-FCL距离检测。

此前机器人对占用图的检测曾把每个link压成一个外接球，例如前臂半径约`0.265 m`。球体会跨入叶片附近的UNKNOWN或桌面体素，从而产生大量假碰撞。代码现在已经改成“原始STL对体素盒”的精确距离计算，AABB只用于快速筛选附近体素，不能再作为最终碰撞几何。

如果同时把桌面或叶片夹具写进机器人URDF、`collision.obstacles`和占用图，会对同一环境重复建模。实验阶段没有必要这样做。

### 1.2 明日修改

- [ ] 拉取包含精确STL—体素碰撞器的新代码。
- [ ] 检查活动URDF/碰撞manifest：只包含ES68各运动link、D435i和安装件。
- [ ] 如果活动URDF中人为加入了桌面link，删除该桌面link。
- [ ] 在`configs/local.yaml`中删除当前`blade_fixture_candidate`静态AABB。
- [ ] 本次实验使用以下配置：

```yaml
collision:
  obstacles: []
  require_obstacles: false
```

这里不是说系统永远不能配置静态障碍物。如果以后存在相机永远看不到、又位于`z>0`运动空间内的固定设施，应当作为显式静态障碍物加入；当前实验台上只有机械臂、叶片和夹具，不需要重复加入。

### 1.3 验收

```bash
uv run bbf robot inspect-model --config configs/local.yaml
```

- [ ] 离线查看六个关节运动时，每个STL随所属link正确运动。
- [ ] 相机和安装件位姿正确。
- [ ] 模型中不存在随机械臂运动的桌面或叶片夹具。
- [ ] 日志中的占用碰撞几何计数表示STL geometry，不再出现按link外接球做最终判定的语义。

停止条件：STL尺度、原点、轴向或相机安装方向错误时，不进行真机运动。

---

## 2. 重测并放宽叶片专用base坐标系包络

### 2.1 为什么修改

当前`configs/local.yaml`中的旧候选值为：

```yaml
proxy_model:
  blade_envelope_min_m: [0.454, 0.001, 0.031]
  blade_envelope_max_m: [0.639, 0.164, 0.368]
  minimum_envelope_retained_fraction: 0.95
```

如果该包络小于真实叶片及鳍片范围，正确的叶片点也会被删除。它会进一步影响：

- 初始代理尺寸；
- 后续自动ROI；
- 粗扫覆盖率；
- PCA/ICP/TSDF；
- 鳍片双面证据；
- schema-5模型。

这个AABB是**叶片科学包络**，只应包含完整叶片和两只鳍片，不应包含夹具、桌面或机器人。

### 2.2 明日修改

- [ ] 固定叶片和夹具，此后不再移动。
- [ ] 使用当前placement的手工首视角hard ROI及已有多视角数据，求叶片点在base坐标系下的XYZ最小值和最大值。
- [ ] 人工确认包络覆盖叶根、叶尖、前缘、后缘、两面以及所有鳍片自由边。
- [ ] 将最终实测值写入：

```yaml
proxy_model:
  blade_envelope_min_m: [<blade_x_min>, <blade_y_min>, <blade_z_min>]
  blade_envelope_max_m: [<blade_x_max>, <blade_y_max>, <blade_z_max>]
  minimum_envelope_retained_fraction: <replayed_fraction_gate>
```

- [ ] 用正确ROI回放，重新计算`minimum_envelope_retained_fraction`，不要直接沿用旧`0.95`。
- [ ] 保存本placement的包络数值、测量来源和回放结果。

不需要为了“安全”无限放大叶片包络；只需保证完整叶片不会被裁掉。包络过大虽然不影响安全占用图，但会把夹具点带入科学模型。

### 2.3 验收

- [ ] `proxy_support_mask.npy`保留完整叶片及鳍片。
- [ ] support overlay中没有明显夹具或桌面点。
- [ ] 初始化输出的保留点比例高于重新计算的门限。
- [ ] 任何被剔除的边界点都经过人工确认不是叶片。

停止条件：完整叶片点被裁掉，或者夹具大面积进入support时，先修正包络，不能继续粗扫。

---

## 3. 按新工作空间重新划分静态自由区并生成验收资产

### 3.1 静态自由空间的含义

占用图由有限视角建立，机器人自身遮挡会留下UNKNOWN。静态自由区只表达以下实验事实：

> 在本次固定工作台布置中，该区域不可能放置叶片、夹具、支架或其他外部物体，因此机器人自遮挡造成的UNKNOWN可以按`accepted_unknown`处理。

它不是：

- 把UNKNOWN改成FREE；
- 忽略OCCUPIED；
- 关闭碰撞检测；
- 运动许可证。

任何OCCUPIED证据仍然优先阻断。

### 3.2 必须先完成第二项

静态自由区必须从完整工作空间中扣除“叶片+夹具目标包络”。因此顺序必须是：

1. 先确定最终叶片包络；
2. 再取得夹具包络；
3. 求二者并集；
4. 最后划分静态自由AABB。

设目标排除包络为：

```text
T_min = componentwise_min(blade_min, fixture_min)
T_max = componentwise_max(blade_max, fixture_max)
```

只考虑工作空间内的`z>=0`部分。按今天确定的工作空间，可使用五个不进入目标包络的AABB：

```yaml
occupancy:
  workspace_bounds_min_m: [-1.00, -0.55, 0.00]
  workspace_bounds_max_m: [ 1.00,  0.55, 1.10]
  accepted_static_free_aabbs:
    - name: base_x_before_target
      minimum_m: [-1.00, -0.55, 0.00]
      maximum_m: [<T_x_min>, 0.55, 1.10]
    - name: base_x_beyond_target
      minimum_m: [<T_x_max>, -0.55, 0.00]
      maximum_m: [1.00, 0.55, 1.10]
    - name: target_column_negative_y
      minimum_m: [<T_x_min>, -0.55, 0.00]
      maximum_m: [<T_x_max>, <T_y_min>, 1.10]
    - name: target_column_positive_y
      minimum_m: [<T_x_min>, <T_y_max>, 0.00]
      maximum_m: [<T_x_max>, 0.55, 1.10]
    - name: above_target_envelope
      minimum_m: [<T_x_min>, <T_y_min>, <T_z_max>]
      maximum_m: [<T_x_max>, <T_y_max>, 1.10]
```

当前local配置使用的目标列边界`x=[0.47,0.63]`、`y=[-0.10,0.14]`已经与旧叶片包络发生重叠：旧叶片本身达到`x=0.639`、`y=0.164`。因此当前静态自由区和旧验收ID都不能继续使用。

### 3.3 重新生成资产

- [ ] 复制模板，不覆盖旧资产：

```bash
cp configs/static_free_acceptance.template.json \
  configs/static_free_acceptance.20260903.completed.json
```

- [ ] 在JSON中写入完全相同的工作空间和五个AABB。
- [ ] 填写真实`operator_id`、UTC时间和检查项。
- [ ] 所有检查项都必须为`true`。
- [ ] 使用新输出目录，例如`es68_d435i_static_free_002`：

```bash
uv run bbf safety record-static-free-acceptance \
  --declaration configs/static_free_acceptance.20260903.completed.json \
  --config configs/local.yaml \
  --output data/acceptance/es68_d435i_static_free_002
```

- [ ] 把命令返回的新路径和新Acceptance ID写入`configs/local.yaml`：

```yaml
occupancy:
  accepted_static_free_acceptance_path: data/acceptance/es68_d435i_static_free_002
  accepted_static_free_acceptance_id: <new_sha256>
```

旧ID：

```text
95db71090e02be1262292d3e1c94adbdf6396de4162e232bb6f66bef556a83e9
```

绑定的是旧工作空间和旧AABB，必须废弃，不能手工保留。

### 3.4 验收

```bash
uv run bbf scan doctor \
  --mode unknown \
  --experimental \
  --config configs/local.yaml
```

- [ ] 静态自由资产的workspace、AABB、机器人几何hash与当前配置一致。
- [ ] 五个静态自由AABB都在工作空间内。
- [ ] 任一静态自由AABB都不与叶片或夹具目标包络相交。
- [ ] OCCUPIED体素即使位于静态自由AABB内仍然阻断。

---

## 4. 保持“全场景安全占用图”和“叶片科学模型”分离

### 4.1 为什么不能共用一个MASK

占用图与粗扫模型使用同一帧FoundationStereo深度，但目的不同：

| 数据产品 | 输入范围 | 用途 |
|---|---|---|
| 安全占用图 | 机器人自遮罩后的全场景有效深度 | 桌面、夹具、叶片和其他障碍的运动安全检查 |
| 叶片科学MASK | 首视角hard ROI，或后续投影引导ROI | 叶片代理、覆盖率、NBV增益、融合和重建 |
| 叶片support点 | 科学MASK再与叶片base AABB求交 | 排除ROI内误画的夹具/桌面点 |

所以：

```text
occupancy_input = integration_valid_mask
blade_science_mask = integration_valid_mask ∩ blade_ROI
blade_support = blade_science_points ∩ blade_base_AABB
```

不能把`blade_science_mask`送给占用建图，否则夹具和其他障碍会从安全地图中消失。

### 4.2 明日工作

该分离逻辑已经在新代码中完成，明日主要负责核对，不需要再设计另一套ROI：

- [ ] 确认占用建图继续读取`integration_valid_mask`。
- [ ] 确认粗扫重建读取`coarse_scan_view/mask.npy`。
- [ ] 确认粗扫累计几何读取`proxy_support_mask.npy`筛选后的点。
- [ ] 不修改占用图，使其只保留叶片。
- [ ] 不把桌面或夹具点送入PCA/ICP/TSDF。

### 4.3 验收

同一视角检查以下资产：

```text
occupancy_mapping/*_integration_valid_mask.npy
coarse_scan_view/mask.npy
coarse_scan_view/proxy_support_mask.npy
coarse_reconstructed_view/base_points_m.npy
```

- [ ] `mask.npy`中的每个像素都来自同帧`integration_valid_mask`。
- [ ] 占用图仍能看见叶片和夹具附近的环境占用。
- [ ] 粗扫support overlay只显示叶片。
- [ ] 两条数据链的source view、sequence、frame number和内容hash一致。

---

## 5. 首视角人工ROI，第二视角起自动生成ROI

### 5.1 首视角

第一正式视角仍由操作员在**整流左图**上绘制一次完整`hard_roi`：

- [ ] 覆盖当前视角所有可见叶片表面和鳍片。
- [ ] 排除夹具、桌面、机器人和背景。
- [ ] 叶片和夹具在绘制后保持不动。

首视角只做确定性多边形栅格化与有效深度求交，不运行最大连通域猜测。

### 5.2 第二视角及以后

后续ROI由程序自动生成，不需要再次人工绘制：

1. 严格读取上一已验收粗扫generation。
2. 累计此前所有叶片support三维点。
3. 使用当前相机位姿和内参投影到整流左图。
4. 将投影区域膨胀，容纳轻微抖动、同步误差、深度空洞和新暴露鳍片。
5. 与当前有效深度求交。
6. 反投影到base坐标系，并与叶片专用AABB求交。

```text
ROI_k = Dilate(Project(accepted_blade_points), radius_px)
        ∩ current_valid_depth
        ∩ current_points_inside_blade_AABB
```

该方法：

- 不复制首视角二维polygon；
- 不选择最大连通域；
- 不因为夹具连通块更大就选择夹具；
- 不删除彼此断开的鳍片区域；
- 投影与当前深度明显不一致时阻断该视角，而不是静默换目标。

### 5.3 明日配置

在eiai的`configs/local.yaml`中同步：

```yaml
bootstrap_foreground:
  projected_reference_dilation_px: 12
  minimum_projected_reference_points: 100
  minimum_projected_reference_pixels: 500
  minimum_projected_match_fraction: 0.50
```

`12 px`是attempt-09真实三视角回放得到的初始值，不是不可改变的物理约束：

| 后续视角 | Precision | Recall | IoU |
|---|---:|---:|---:|
| 第二视角 | 89.79% | 100.00% | 89.79% |
| 第三/背面视角 | 98.89% | 98.35% | 97.28% |

### 5.4 验收

- [ ] 第一视角`foreground.algorithm`为人工bootstrap算法并绑定hard ROI。
- [ ] 第二视角起`foreground.algorithm`为`accumulated_blade_projection_base_envelope_v1`。
- [ ] 自动视角的`coarse_scan_view/metadata.json`为schema 3。
- [ ] metadata包含`foreground_reference_generation`及其`generation.json`哈希。
- [ ] 自动MASK覆盖新暴露的鳍片边缘，没有明显夹具/桌面区域。
- [ ] 自动算法失败时保存该帧诊断，不现场恢复“最大连通域”逻辑。

---

## 6. 明日推荐执行顺序

必须按依赖关系执行：

1. [ ] 拉取代码并备份eiai现有`configs/local.yaml`。
2. [ ] 完成第一项：删除重复桌面/夹具碰撞模型，检查原始STL装配。
3. [ ] 完成第二项：测量并写入最终叶片AABB及保留比例。
4. [ ] 根据最终叶片AABB和夹具AABB计算目标排除包络。
5. [ ] 完成第三项：重写五个静态自由AABB，生成新验收资产和新ID。
6. [ ] 同步第五项的四个投影ROI参数。
7. [ ] 运行unknown doctor，确保没有配置/资产不一致。
8. [ ] 使用保存数据做不运动回放，分别检查占用MASK、叶片MASK和support点。
9. [ ] 低速运行一个首视角和一个自动第二视角，检查schema-3自动MASK。
10. [ ] 第二视角验收通过后，再继续完整自动粗扫。

不要在同一次试验中一边运动一边继续修改AABB、ROI参数或验收资产。任一配置变化都应使用新的run输出；叶片或夹具发生物理移动时，还必须使用新的`placement_id`。

## 7. 明日完成标准

五项同时满足才算完成：

- [ ] 机械臂碰撞使用原始STL，不再受大外接球假碰撞影响。
- [ ] 叶片科学包络覆盖完整双面鳍片，但不包含夹具和桌面。
- [ ] 静态自由区完全避开叶片/夹具，并有匹配当前工作空间的新验收ID。
- [ ] 安全占用图保留全场景，科学重建只消费叶片点。
- [ ] 只在第一视角人工画ROI，第二视角起能够自动生成并回放验证叶片MASK。
