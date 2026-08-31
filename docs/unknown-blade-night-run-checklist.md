# 未知叶片今晚完整真机运行清单

适用目标：人工三视角启动 → 粗扫 → schema-5 → 精扫 → 最终模型。

本清单按实验模式编写。实验模式保留全部运动安全门，但跳过尚未完成的科学与运行时发布验收，
最终结果只能作为实验数据，不能声明为生产验收结果。

公式约定：\(T^a_b\)表示“把\(b\)坐标系中的点变换到\(a\)坐标系”；\(R\)和\(\mathbf t\)
分别表示旋转与平移，\(\mathbf q\)表示六关节向量，\(\mathbf c\)表示相机光心，\(\mathbf n\)表示
单位表面法向。所有长度默认使用米，关节角默认使用弧度；带`deg`的配置与公式明确使用度。

## 0. 三个不可违反的身份规则

- `placement_id`表示夹具和叶片一次不变的真实物理放置。
- `run_id`表示该放置下的一次软件运行或重试。
- `output`表示该次运行的独立、不可覆盖实验目录。

今晚第一次放置可使用：

```text
placement_id = blade-placement-20260831-01
run_id       = blade-placement-20260831-01-attempt-01
output       = data/experiments/blade-placement-20260831-01-attempt-01
```

同一物理位置的软件失败重试：保留`placement_id`，把attempt改为02并使用全新output。夹具或叶片
发生任何移动：停止当前实验，创建`blade-placement-20260831-02`，旧地图、粗模型、schema-5、
精扫覆盖和permit全部不得复用。

### 数学原理：为什么必须区分三个身份

把一次真实放置记为物理状态\(P\)，一次软件执行记为\(R\)，输出资产记为\(A\)。系统要求：

$$
A = F(P,R;\ \text{代码、配置、标定、模型、原始帧})
$$

`placement_id`固定\(P\)，`run_id`区分同一个\(P\)上的不同执行\(R\)。如果实物移动后仍沿用原
`placement_id`，就等于错误地断言新旧观测来自同一个静态坐标系；旧占用图、粗表面和精扫覆盖
中的点将被当成仍在原位置，这不是普通重试，而是坐标系前提被破坏。

每个运行事件还形成SHA-256链：

$$
H_i=\operatorname{SHA256}\!\left(\operatorname{canonical}(E_i,H_{i-1})\right).
$$

因此修改任一历史事件\(E_i\)都会改变从\(H_i\)开始的全部后继哈希。全新且不可覆盖的`output`
保留了这个单向证据链，也避免两个attempt的资产发生多对一覆盖。

## 1. 放置前安全边界

- [ ] 现场至少一人始终能直接触及物理急停。
- [ ] 机器人速度缩放保持local配置中的`0.08`，不临时提高。
- [ ] D435i、末端连接件、机器人底座和工作台均未移动。
- [ ] 夹具和叶片在整个实验中刚性固定，实验开始后不再触碰。
- [ ] 整个叶片和夹具完全位于静态自由验收留下的目标包络内：

```text
x = 0.47 ～ 0.63 m
y = -0.10 ～ 0.14 m
z = 0.00 ～ 0.37 m
```

- [ ] 夹具本体仍位于配置的保守碰撞AABB内：

```text
x = 0.48 ～ 0.62 m
y = -0.095 ～ 0.135 m
z = -0.02 ～ 0.06 m
```

任一项不成立，不运行运动入口。重新测量并更新验收资产，不能仅凭“移动很小”继续。

### 数学原理：包络、膨胀碰撞体与静态场景假设

轴对齐包围盒（AABB）由下界\(\boldsymbol\ell\)和上界\(\mathbf u\)定义。任一点
\(\mathbf p=(x,y,z)^\mathsf T\)在盒内，当且仅当：

$$
\boldsymbol\ell\leq\mathbf p\leq\mathbf u
$$

三个分量都成立。这里要求的是**整个**叶片/夹具点集\(S\)满足
\(\forall\mathbf p\in S\)，而不是只要求中心点在盒内。相机、机器人连杆和障碍物的安全检查还会
按配置间隙做保守膨胀；几何上等价于障碍物与半径为\(r\)的球做Minkowski和
\(O\oplus B_r\)。只要机器人几何与这个膨胀体相交，就不允许运动。

占用图采用静态世界假设：同一次placement内，世界几何不随采集帧改变。叶片或夹具移动会使
旧射线证据\((\mathbf o_i,\mathbf p_i)\)失效，所以即使移动只有几毫米，也必须创建新placement，
不能用软件阈值掩盖物理变化。

## 2. eiai代码和环境准备

在项目根目录执行：

```bash
git status --short
git pull --ff-only origin main
git status --short
git log -1 --oneline
```

- [ ] 第一次`git status`没有不明修改；如有修改先人工确认归属。
- [ ] pull成功且没有合并提交。
- [ ] pull后的`git status`为空。
- [ ] 最新提交包含正式`placement_id`功能。

安装/核对目标依赖：

```bash
uv sync --extra foundation-stereo --extra tsdf-open3d
uv run python -c "import open3d; print(open3d.__version__)"
uv run bbf camera list
uv run bbf stereo doctor --config configs/local.yaml
```

- [ ] Open3D可以导入。
- [ ] 只出现目标D435i序列号`243222074585`。
- [ ] FoundationStereo源码、配置、权重、Torch和CUDA检查通过。

### 数学原理：可复现运行是一个确定输入集合

把重建结果写成：

$$
Y=F(C,G,K,W,D),
$$

其中\(C\)是代码提交，\(G\)是配置，\(K\)是内外参与手眼标定，\(W\)是模型权重，\(D\)是
原始数据。`git log`、配置快照、资产哈希和原始session共同固定这些输入。任何一项不同，得到的
\(Y\)都不再是同一条可回放实验链。

双目整流的目标是让对应点落在同一图像行，即理想情况下\(v_l-v_r=0\)。垂直视差、标定残差或
错误相机序列号会破坏这个条件；而深度\(Z=f_xB/d\)对视差误差很敏感：

$$
\frac{\delta Z}{Z}\approx-\frac{\delta d}{d}.
$$

所以依赖、相机和标定检查不是环境整理步骤，而是后续几何精度成立的前提。

## 3. local配置核对

不得覆盖Git中的`configs/default.yaml`。今晚只使用eiai的`configs/local.yaml`。

确认以下值：

```yaml
robot:
  robot_ip: 192.168.6.60
  local_ip: 192.168.6.61
  motion_enabled: true

realsense:
  serial_number: "243222074585"
  infrared_emitter_enabled: true

stop_and_capture:
  enabled: true
  maximum_segment_joint_delta_rad: 0.02

occupancy:
  enabled: true
  integration_stride: 2
  maximum_map_age_s: null
  minimum_source_views: 3
  maximum_source_views: 3
  minimum_free_observations: 3
```

运行解析检查：

```bash
uv run python -c "from biblade_fusion.core.settings import load_settings; s=load_settings('configs/local.yaml'); print(s.occupancy.maximum_map_age_s, s.occupancy.minimum_source_views, s.occupancy.maximum_source_views, s.occupancy.minimum_free_observations)"
```

必须输出：

```text
None 3 3 3
```

禁止在实验开始后修改local配置、标定、模型、STL、FoundationStereo权重或多边形。

### 数学原理：三视图地图与“没有5秒寿命”

当前安全地图最多保留3个来源；`MAP_READY`时必须恰好有3个：

$$
|S_k|\leq3,\qquad |S_k|=3\ \text{when MAP_READY},\qquad
S_{k+1}=\operatorname{tail}_3\!\left(\operatorname{independent}(S_k\cup\{v_{k+1}\})\right).
$$

新帧成功后，系统用保留下来的独立来源加新帧**重新构建一个不可变generation**；它不是给旧
体素图设置5秒倒计时，也不是把历史所有帧无限累加。若新帧与某个旧视角过近，冲突的旧来源被
替换，而不是把重复帧算作新的FREE证据。新generation完整构建并发布后才替换当前generation。

`maximum_map_age_s: null`的数学含义是不存在
\(t_{now}-t_{map}\leq5\text{s}\)这一墙钟判据。当前`MAP_READY` generation持续有效，直到后续
成功采集产生并发布下一generation，或者发生mapping context/placement身份变化、证据不一致等显式
失效事件。普通的每段停稳采集本身不会让地图按时间失效。

每个体素\(x\)是三态变量：

$$
M(x)=
\begin{cases}
\text{OCCUPIED}, & n_{occ}(x)\geq1,\\
\text{FREE}, & n_{occ}(x)=0\ \land\ n_{free}(x)\geq3,\\
\text{UNKNOWN}, & \text{其他情况}.
\end{cases}
$$

因此`minimum_source_views = maximum_source_views = minimum_free_observations = 3`彼此配套：只有三个
独立视角都把某体素作为射线穿越空间，它才成为FREE；一次占用端点证据优先，UNKNOWN从不被
猜成FREE。

## 4. 新放置的第一帧预览与多边形

### 4.1 将机器人放到第一安全视角

- [ ] 用示教器把相机放到已知安全、能看到完整可见叶片和鳍片的正面视角。
- [ ] 确认电缆无拉扯、镜头无遮挡、叶片没有超出图像边界。
- [ ] 停止机器人程序，使控制器最终报告`robot_mode=IDLE`。
- [ ] 从此刻到正式第一帧采集完成，不移动机器人、相机、夹具或叶片。

数学原理：初始代理面的法向来自点云协方差的最小特征向量。正视程度为
\(|\mathbf n^\mathsf T\mathbf v|\)，其中\(\mathbf n\)是估计法向，\(\mathbf v\)是表面指向相机的
单位向量。过于擦边的视角使这个余弦接近0，深度噪声会被放大，法向符号和隐藏面的推断也不稳定。
完整入镜则保证后面PCA包络不是对被裁掉叶片的错误包络。

### 4.2 预采集一份同步session

```bash
uv run bbf acquire snapshot \
  --config configs/local.yaml \
  --view-id blade-placement-20260831-01-preview
```

记录命令打印的session绝对路径。下面用`<PREVIEW_SESSION>`表示该路径。

数学原理：采集不是把“某张图”和“稍后读到的关节角”随意配对。系统用时间戳选择与双目曝光
最接近的机器人状态，并验证停稳与时序边界。若关节向量为\(\mathbf q\)，正运动学给出
\(T^b_f(\mathbf q)\)，整流左相机在base中的位姿为：

$$
T^b_{c_r}=T^b_f(\mathbf q)\,T^f_{l}\,\left(T^{c_r}_{l}\right)^{-1},
$$

其中\(T^f_l\)是flange到原始左红外相机的手眼标定，\(T^{c_r}_l\)是整流相机到原始左相机的
标定关系。矩阵乘法顺序不可交换。

### 4.3 对预览session运行FoundationStereo

输出目录必须不存在：

```bash
uv run bbf stereo infer-session \
  --config configs/local.yaml \
  --session <PREVIEW_SESSION> \
  --view-id blade-placement-20260831-01-preview \
  --output data/placements/blade-placement-20260831-01/preview-stereo
```

导出整流左图用于标注：

```bash
uv run python -c "import cv2, numpy as np; a=np.load('data/placements/blade-placement-20260831-01/preview-stereo/left_rectified.npy'); assert cv2.imwrite('data/placements/blade-placement-20260831-01/left_rectified.png', a)"
```

数学原理：整流后，同一三维点在左右图的横向视差为\(d=u_l-u_r\)。针孔双目模型给出：

$$
Z=\frac{f_xB}{d},\qquad
X=\frac{(u-c_x)Z}{f_x},\qquad
Y=\frac{(v-c_y)Z}{f_y}.
$$

只接受有限且\(d>0\)的像素。左右一致性会在右图对应位置\(u_r\approx u_l-d_l\)读取右视差，
计算\(e_{LR}=|d_l-d_r|\)，并使用：

$$
c=\exp\!\left(-\frac{e_{LR}}{\tau_{LR}}\right).
$$

这里的\(c\)是确定性的左右一致性分数，不是经过概率标定的“正确率”。误差超过阈值的像素不
进入有效深度；安全占用还要求\(c\)通过`minimum_stereo_confidence`。

### 4.4 手动画完整`hard_roi`

在1280×720的`left_rectified.png`上记录多边形顶点，按轮廓顺序写入：

```text
data/placements/blade-placement-20260831-01/bootstrap_polygon.json
```

格式：

```json
{
  "vertices_uv": [
    [u0, v0],
    [u1, v1],
    [u2, v2]
  ]
}
```

- [ ] 顶点全部位于`0≤u<1280, 0≤v<720`。
- [ ] 多边形不自交，沿顺时针或逆时针连续排列。
- [ ] 包含全部可见主表面、鳍片、叶根、叶尖、前缘和后缘。
- [ ] 排除夹具、支架、机器人和背景。
- [ ] 不使用005的小表面seed作为完整hard ROI。

数学原理：多边形顶点由人手工给出，程序只做确定性栅格化。设多边形内部指示函数为
\(I_P(u,v)\)，FoundationStereo有效深度指示为\(I_D(u,v)\)，则hard ROI的最终前景是：

$$
M_{hard}(u,v)=I_P(u,v)\land I_D(u,v).
$$

因此它不会按颜色、最大连通域或学习模型自动“猜”叶片，也不会删除多边形内与主表面断开的
细鳍片。代价是：画进夹具的有效深度也会被当作叶片，所以轮廓必须由人确认。多边形使用的是
**整流左图像素坐标**，不是原始左图、RGB图或三维坐标。

### 4.5 离线验证多边形

输出目录必须不存在：

```bash
uv run bbf scan bootstrap-mask \
  --config configs/local.yaml \
  --stereo data/placements/blade-placement-20260831-01/preview-stereo \
  --polygon data/placements/blade-placement-20260831-01/bootstrap_polygon.json \
  --seed-mode hard_roi \
  --output data/placements/blade-placement-20260831-01/bootstrap-mask-check
```

- [ ] 命令成功并打印`mask_pixels`和`mask_fraction`。
- [ ] 人工查看`mask.npy`或叠加图，确认完整叶片被保留且夹具未进入。
- [ ] 若失败或掩模错误：修改多边形，并使用新的检查输出目录；不得覆盖旧检查资产。
- [ ] 验证完成后机器人和实物仍未移动。

数学原理：除hard ROI交集外，验证器仍执行总量与边界门槛。掩模比例为

$$
r_M=\frac{\sum_{u,v}M_{hard}(u,v)}{H\,W}.
$$

像素数、\(r_M\)、有效seed比例必须落在配置范围内；掩模若触及有效深度域边界则失败，因为此时
无法区分“物体确实到边界”和“物体已被深度有效域截断”。输入图、深度、有效掩模、多边形策略
都写入内容哈希，所以修改顶点后必须产生新的检查资产。

## 5. 非运动完整doctor

本次配置的科学和运行时发布验收为null，所以必须明确使用实验审计：

```bash
uv run bbf scan doctor \
  --mode unknown \
  --experimental \
  --config configs/local.yaml
```

- [ ] 命令返回0。
- [ ] 无FAIL项目。
- [ ] 明确理解实验结果不具有生产科学验收身份。

doctor失败时不启动机器人。保存完整输出，修复根因后重新运行doctor；不能只删检查或放宽阈值。

### 数学原理：doctor是合取门，不是平均分

若各检查为布尔门\(g_1,\ldots,g_n\)，总结果是：

$$
G=\bigwedge_{i=1}^{n}g_i.
$$

任何一项FAIL都会使\(G=0\)，不存在“其他项目很好所以抵消一个FAIL”。`--experimental`只明确
承认当前science/runtime acceptance尚未发布，并不会把机器人、相机、标定、碰撞、占用或停止
协议门改成可忽略。缺失、异常或无法证明统一按不通过处理，即fail closed。

## 6. 正式运行前最后检查

- [ ] `data/experiments/blade-placement-20260831-01-attempt-01`不存在。
- [ ] 多边形文件已保存且不会再修改。
- [ ] 机器人仍在与预览相同的第一视角。
- [ ] 控制器为IDLE，安全状态正常，急停可触及。
- [ ] 没有人位于机器人工作空间内。
- [ ] 另一个终端已准备好，只用于只读监督。

正式命令：

```bash
uv run bbf scan run-unknown \
  --experimental \
  --config configs/local.yaml \
  --placement-id blade-placement-20260831-01 \
  --run-id blade-placement-20260831-01-attempt-01 \
  --output data/experiments/blade-placement-20260831-01-attempt-01 \
  --operator-id vale \
  --bootstrap-polygon data/placements/blade-placement-20260831-01/bootstrap_polygon.json \
  --bootstrap-seed-mode hard_roi \
  --first-side front
```

启动阶段会在连接硬件前预加载FoundationStereo。等待命令明确进入控制台，不要重复启动第二实例。

只读监督终端：

```bash
uv run bbf supervise replay \
  --snapshot data/experiments/blade-placement-20260831-01-attempt-01/live_timeline \
  --follow
```

### 数学原理：冻结输入后才有可审计因果链

正式开始时系统把`placement_id`、`run_id`、配置、标定、模型、polygon和代码身份绑定进初始事件。
后续每一代资产都是纯追加关系：

$$
A_{k+1}=F(A_k,D_{k+1},G),
$$

其中\(G\)是在run开始时冻结的策略集合。运行中修改\(G\)会让前后generation不再服从同一个
函数，历史质量数值也无法比较。要求output不存在，是为了保证\(A_0\)唯一且没有旧文件混入。
FoundationStereo先预加载再连接运动硬件，是为了在任何运动授权前暴露模型/CUDA初始化失败。

## 7. 人工三视角启动

### 第一视角

控制台显示`NEEDS_CAPTURE`且没有预定view ID时：

- [ ] 再次确认控制器IDLE。
- [ ] 输入一次且仅一次：`c`。
- [ ] 等待stop、停稳、采集、FoundationStereo、自遮罩、占用重建和科学资产全部返回。
- [ ] 不因计算时间较长而再次输入`c`、移动机器人或关闭终端。

第一帧会使用命令绑定的hard ROI及`front`侧标记。地图通常仍为`MAPPING`。

第一帧的三维点先由双目反投影得到\(\mathbf p^{c_r}\)，再变换到base：

$$
\mathbf p^b=R^b_{c_r}\mathbf p^{c_r}+\mathbf t^b_{c_r}.
$$

安全占用对每条有效射线从相机中心\(\mathbf o\)走到深度端点\(\mathbf p\)。端点体素投
OCCUPIED票，\(\mathbf o\)到\(\mathbf p-\epsilon\hat{\mathbf r}\)之间的体素投FREE票；
\(\epsilon\)是`free_space_margin_m`，用于避免把表面前的离散误差误清为空闲。第一帧只有一张
独立FREE票，所以不可能达到三票，保持`MAPPING`是正确结果。

机器人自遮罩使用关节角和碰撞几何渲染预测机器人深度\(D_r(u,v)\)。测量深度落在配置的前/后
容差带并经过像素膨胀时标记为机器人；这些像素不参与环境射线积分。无效、低置信度或被遮罩像素
保持UNKNOWN，绝不会生成清空射线。

### 第二视角

- [ ] 确认控制台重新提示人工重定位。
- [ ] 使用示教器把已停止机器人移动到第二个已知安全视角。
- [ ] 相机光心相对所有已接受视角至少平移2厘米，或观察方向至少变化5度。
- [ ] 保持叶片和夹具完全不动。
- [ ] 结束示教运动并确认控制器IDLE。
- [ ] 输入一次`c`，等待完整周期完成。

数学原理：两个相机视角\(i,j\)独立，当且仅当至少满足一个条件：

$$
\|\mathbf c_i-\mathbf c_j\|_2\geq0.02\ \text{m}
\quad\lor\quad
\theta_{ij}=\cos^{-1}\!\left(\operatorname{clip}(\mathbf a_i^\mathsf T\mathbf a_j,-1,1)\right)
\geq5^\circ,
$$

其中\(\mathbf c\)是整流左相机光心，\(\mathbf a\)是相机+Z光轴，二者都在base坐标系中。
这是几何独立性，不是“view ID不同”或“等待了一段时间”。

### 第三视角

重复第二视角流程，第三视角也必须与前两个视角分别满足独立性。建议三个视角具有明显不同的
相机方向，且至少为粗模型提供两侧证据；不能只在几乎相同姿态连续拍三次。

- [ ] 第三次输入`c`后等待地图达到`MAP_READY`。
- [ ] 若仍为`MAPPING`，查看阻塞原因；不得把minimum值改小或把UNKNOWN改成FREE。

安全地图只保留最近三个独立来源，但粗扫和精扫科学覆盖会持续累计。当前MAP_READY地图不会按
墙钟时间过期；只有下一次成功发布的新generation会替换它。

第三帧后，对任一体素必须同时有三张独立视图的FREE射线票才可成为FREE。OCCUPIED采用
“occupied wins”：

$$
O_{k+1}=O_k\cup O_{new},\qquad
F_{k+1}=\{x:n_{free}(x)\geq3\}\setminus O_{k+1}.
$$

一张后来的FREE射线不能擦除同一次三来源重建中的已有OCCUPIED证据；当最旧来源离开滑动窗口时，
下一generation会从仍保留的来源重新构建，所以这里不是跨所有时间永久累积OCCUPIED。
`MAP_READY`只表示来源数量及自遮罩、深度质量、标定一致性和证据链门槛全部通过，不表示所有空间
都已知。UNKNOWN默认在运动碰撞查询中保守阻塞；唯一例外是落在预先验收、哈希绑定且与当前
mapping context一致的静态自由AABB内，此时会作为`accepted_unknown`单独审计，绝不是把地图中的
UNKNOWN改写成FREE。

这里要再次区分两类“累计”：安全地图只在滑动的三来源集合上逐帧生成新generation；粗扫/精扫
科学账本则按唯一观测ID持续累计，直至对应阶段结束。两者不能互相替代。

## 8. 自动粗扫与逐段批准

MAP_READY后，系统自动创建代理模型、普通法向候选和每侧±15°鳍片发现候选。操作员不手工指定
科学view ID。

### 8.1 数学原理：从第一面构造保守双侧代理

对体素降采样后的初始点\(\mathbf p_i\)计算质心和协方差：

$$
\bar{\mathbf p}=\frac1N\sum_i\mathbf p_i,\qquad
C=\frac1N\sum_i(\mathbf p_i-\bar{\mathbf p})(\mathbf p_i-\bar{\mathbf p})^\mathsf T.
$$

对\(C\)做特征分解。最大特征值方向作为叶片主轴，最小特征值方向作为表面法向；法向符号由
相机所在一侧确定，第三轴用叉乘构成右手正交坐标系。PCA只能从首面看见的点估计切向范围，不能
从单面观测恢复背面，所以代码必须使用配置的`estimated_thickness_m`，再加可见侧、隐藏侧和切向
margin形成保守双侧OBB。这也是厚度参数不能由单帧“自动准确推断”的原因。

### 8.2 数学原理：普通视点、覆盖网格与斜视发现

针孔相机在距离\(s\)处的可用足迹近似为：

$$
w=2s\tan\frac{\phi_x}{2}\,\eta,\qquad
h=2s\tan\frac{\phi_y}{2}\,\eta,
$$

其中\(\phi_x,\phi_y\)由内参和非居中主点计算，\(\eta\)是`footprint_utilization`。相邻目标的
步长是\(w(1-o)\)、\(h(1-o)\)，\(o\)为重叠率；覆盖长度\(L\)所需格数为：

$$
n=\begin{cases}
1,&L\leq w,\\
\left\lceil\dfrac{L-w}{w(1-o)}\right\rceil+1,&L>w.
\end{cases}
$$

每个普通候选把相机放在patch中心\(\mathbf x\)外法向\(\mathbf n\)的standoff处：
\(\mathbf c=\mathbf x+s\mathbf n\)，相机+Z轴朝向\(-\mathbf n\)。鳍片未知时，系统在每一侧、
两个面内轴上各生成±15°成对斜视：

$$
\mathbf c_{\pm}=\mathbf x+s\cos\alpha\,\mathbf n
\pm s\sin\alpha\,\mathbf t,\qquad \alpha=15^\circ.
$$

正负斜视从相反方向观察凸起薄片；major/minor两组提供正交的确定性备选，避免预先假定鳍片延伸
方向。

### 8.3 数学原理：候选不是运动许可

候选的几何量包括：

$$
c_{look}=\mathbf z_c^\mathsf T
\frac{\mathbf x-\mathbf c}{\|\mathbf x-\mathbf c\|},\qquad
c_{inc}=(-\mathbf z_c)^\mathsf T\mathbf n,
$$

以及standoff误差、足迹覆盖率、相机到代理的间隙和workspace/AABB关系。几何评分仅用于比较；
必须另外通过workspace、禁入体、IK以及用FK回算的端点残差。即便端点可达，也只证明存在一个
关节解\(\mathbf q_g\)，没有证明从当前\(\mathbf q_0\)到\(\mathbf q_g\)的整条轨迹安全。

每次控制台显示：

```text
Prepared one segment. Exact approval token: EXECUTE ...
```

逐段执行：

- [ ] 观察只读GUI中的目标、计划短段、机器人和占用图。
- [ ] 确认现场没有新增人员或障碍物，电缆状态正常。
- [ ] 确认计划运动方向符合预期。
- [ ] 仅在确认安全后，原样复制完整`EXECUTE ...`字符串并回车。
- [ ] 不手工缩短、重写或复用上一次token。
- [ ] 保持手在物理急停附近，观察整个短段。
- [ ] 每段结束后等待自动stop、停稳和采集完成。

### 8.4 数学原理：等分短段、连续碰撞证明与一次性token

设剩余目标关节差\(\Delta\mathbf q=\mathbf q_f-\mathbf q_0\)，单段上界为\(b=0.02\) rad。系统取：

$$
N=\max\!\left(1,\left\lceil
\frac{\|\Delta\mathbf q\|_\infty}{b}\right\rceil\right),\qquad
\mathbf q_1=\mathbf q_0+\frac1N\Delta\mathbf q.
$$

每次只准备这一个等分短段。这样所有段都有相近且不超过\(b\)的关节变化，避免“先走满步、最后
剩一个极短段”，因为极短段结束后的强制采集可能无法构成独立占用视角。

待证明路径是关节空间直线：

$$
\mathbf q(s)=\mathbf q_0+s(\mathbf q_1-\mathbf q_0),\qquad s\in[0,1].
$$

系统不只抽样若干点后假定中间安全。它对路径区间递归二分，在区间中点计算网格/网格间隙或
机器人包围球到占用体素的关系，再用该半区间最大关节变化推导几何最大位移\(B\)。只有

$$
d_{mid}-B-\varepsilon>0
$$

时整个区间才被连续认证；否则继续细分。出现碰撞见证、模型错误、达到细分深度仍不能证明，结果
都不是CLEAR，运动被阻止。占用UNKNOWN只有在预先验收且哈希绑定的静态自由AABB内才可记为
`accepted_unknown`；其余UNKNOWN仍阻塞。网格碰撞证明和占用碰撞证明必须同时有效。

批准token绑定路径端点、当前占用generation序号和内容哈希、质量证据、策略、机器人/碰撞几何、
运行时运动合同、stop generation及单调时钟有效期。可概括为：

$$
T=\operatorname{SHA256}(\mathbf q_0,\mathbf q_1,H_{map},H_{robot},H_{policy},g_{stop},t_{exp}).
$$

任一输入改变，旧token都不匹配；token被消费一次或过期后不能复用。

较长路径会拆成多个`TRANSIT`段。TRANSIT采集只更新安全地图，不增加科学覆盖；系统会保持同一
最终科学目标。不要因为连续出现多个相似目标而中止或手工换目标。

粗扫不会在“看起来差不多”时结束。必须同时满足：

- [ ] 总粗扫视图数至少6。
- [ ] 正反侧各至少3。
- [ ] 每侧至少一对验证过的相反斜视图。
- [ ] 代理表面覆盖完成。
- [ ] 两侧鳍片都具有两个物理表面证据。

### 8.5 数学原理：粗扫为什么不能靠“看起来完整”结束

代理面被分成patch，每个patch再分成\(B\times B\)小格。小格点数为\(n_{ij}\)，patch覆盖率是：

$$
C_p=\frac1{B^2}\sum_{i,j}
\mathbf1[n_{ij}\geq n_{min}].
$$

仅当\(C_p\)达到`coverage.completed_fraction`才算该patch完成；全代理覆盖要求所有正反patch都
完成。同一物理帧有稳定观测ID，重复读盘不能重复加票。

粗扫终止是多个硬条件的合取：

$$
G_{coarse}=G_{views\geq6}\land G_{front\geq3}\land G_{back\geq3}
\land G_{coverage}\land G_{oblique\ pairs}\land G_{fin\ two\ faces}.
$$

“相反斜视对”要求该侧同一轴的正负候选都实际采集且位姿误差在门槛内；只规划出来或IK可达不
计数。鳍片`two_faces_observed`要求融合几何中确实存在分离的两个物理面证据，不能由配置厚度、
镜像代理或单面点云推测出来。

## 9. schema-5交接

粗扫门槛全部通过后系统自动：

1. 固化最终粗扫checkpoint；
2. 构建并严格重读schema-5；
3. 记录`handoff_prepared`；
4. 停止粗扫协调器；
5. 创建独立精扫协调器、占用publisher和安全工厂；
6. 写入精扫启动事件；
7. 在当前停止位姿执行`fine_transition_bootstrap_000`。

- [ ] 交接期间不移动机器人或叶片。
- [ ] 不启动第二个程序。
- [ ] 不把长时间计算误认为死机；观察日志和GPU利用率。
- [ ] 精扫若提示安全来源不足，只在明确提示时输入一次`c`，且不带front/back。

粗扫permit、批准、prepared segment和旧地图publication不会进入精扫；这是正常现象。

### 数学原理：schema-5是固定参考面的原子交接

粗扫输出不是“当前内存中的一个点云指针”，而是由精确来源序列、配置和生成物哈希绑定的不可变
参考\(S_5\)：

$$
H_{S_5}=\operatorname{SHA256}
(H_{views},H_{fusion},H_{surface},H_{plan},H_{TSDF},H_{quality}).
$$

写完后立即用严格reader重读，相当于验证序列化结果仍满足schema、形状、单位、枚举、内容哈希和
来源约束；只有重读成功，`handoff_prepared`才进入顶层事件链。精扫后续所有coverage generation
都引用同一个\(H_{S_5}\)，不允许随着新帧悄悄改变目标面，否则“覆盖率提高”将失去共同分母。

交接采用两个独立状态机。粗扫的permit绑定粗扫协调器状态\(X_c\)，精扫从新状态\(X_f^0\)开始，
且\(X_c\)中的permit、prepared segment和地图publication都不属于\(X_f^0\)。形式上没有
\(\operatorname{permit}(X_c)\to\operatorname{permit}(X_f^0)\)的继承映射。精扫必须在当前停止位
重新采集并建立自己的三来源安全图，这正是bootstrap帧存在的原因。

整个交接事件仍遵守\(H_i=\operatorname{SHA256}(E_i,H_{i-1})\)。因此出现schema-5但没有对应
handoff/fine-start事件时，不能人工拼接目录宣布精扫已开始。

## 10. 精扫循环

精扫以schema-5为固定参考，逐项覆盖正反主表面、前后缘、叶根、叶尖、鳍片面、鳍根和自由边。

### 10.1 数学原理：曲面patch与自适应视点

schema-5曲面把每个patch固定为参考采样点\(\mathbf x_j\)、法向\(\mathbf n_j\)、区域类型和OBB。
候选相机中心沿patch主法向放置：

$$
\mathbf c=\mathbf x_{obb}+s\mathbf n,
$$

相机朝向patch。高曲率以及前后缘、根、尖、鳍根和鳍自由边优先尝试更近的合法standoff；普通
区域优先尝试最接近基准距离的值。每个距离都要通过投影覆盖率和z-buffer可见率。若整块不可见，
程序在有限深度内把patch细分再规划；超过细分上限仍无可见视点则失败，而不是生成盲视点。

投影仍用针孔模型：

$$
u=f_xX/Z+c_x,\qquad v=f_yY/Z+c_y.
$$

多个参考点落到同一像素时只保留最小正深度\(Z\)，即z-buffer。目标patch只有在自己的深度与全
表面最近深度一致时才拥有该像素，这能阻止薄壁背面的patch被前表面错误计为可见。

每个精扫目标仍执行与粗扫完全相同的短段批准流程：

- [ ] 检查GUI和现场。
- [ ] 原样粘贴当前唯一token。
- [ ] 观察短段并随时准备物理急停。
- [ ] 等待自动stop、停稳、采集、参考投影掩模、覆盖和质量评估完成。

### 10.2 数学原理：schema-5投影掩模不是再次手画

精扫不再使用初始polygon决定每一帧。固定参考表面投影得到预测深度\(D_s(u,v)\)，当前帧测量为
\(D(u,v)\)，eligible包含双目有效性、置信度和其他前置质量门。接受掩模为：

$$
M(u,v)=I_{eligible}\land I_{zbuf}\land
\big[D_s-\tau_f\leq D\leq D_s+\tau_b\big].
$$

前后容差\(\tau_f,\tau_b\)可以不对称。目标patch还必须满足入射余弦、投影支持像素数、目标匹配
像素数、全参考匹配率和目标匹配率。这样背景、遮挡物和另一薄壁面不能仅因靠得近就进入科学点云。

### 10.3 数学原理：精扫覆盖和质量如何更新

对于固定参考点\(\mathbf s_j\)，从本侧本帧点云中找最近点\(\mathbf p_{k(j)}\)，记录跨全部有效
观测的最小距离：

$$
d_j=\min_k\|\mathbf s_j-\mathbf p_k\|_2.
$$

只有相机位于正确侧且入射余弦通过时，该帧才更新对应侧patch。定义已观察集合
\(O=\{j:d_j\leq d_{max}\}\)，则：

$$
C=\frac{|O|}{N},\qquad
\operatorname{RMSE}=\sqrt{\frac1{|O|}\sum_{j\in O}d_j^2},\qquad
N_c=\frac1{|O|}\sum_{j\in O}|\mathbf n_j^\mathsf T\hat{\mathbf n}_{k(j)}|.
$$

patch完成要求观测点数、\(C\)、RMSE和法向一致性\(N_c\)全部过门。绝对点数要求取配置值与
patch实际参考点数的较小者，避免细分后的微小patch出现数学上不可能达到的点数门槛。

### 10.4 数学原理：下一视点排序与重采上限

先按配置的区域优先级排序，再依次偏向覆盖缺口\(1-C\)更大、法向缺口\(1-N_c\)更大、
RMSE相对门槛更差、可见率/投影率/几何评分更高的候选。它是字典序，不是把不同量纲随意加权成
一个总分；关节最大/总行程只作为末级tie-break，不得把“近但科学价值低”的视点提前。

已采过而仍不合格的patch只生成配置中有限的确定性扰动：每个扰动固定距离偏置、倾角和方位角，
并重新通过几何、workspace、IK、FK和连续路径证明。最多三次意味着候选集合是有限集合
\(\{v_1,v_2,v_3\}\)，耗尽后必须BLOCKED，不能随机游走或无限重拍。

同一patch质量不足时，系统只使用配置中三组确定性重采扰动，不允许无限尝试。没有候选通过
workspace、IK、FK或连续安全证明时会BLOCKED；不要现场放宽阈值继续运动。

TRANSIT帧只更新安全generation，不写入上面的科学账本；只有到达精确科学候选、完成该候选的
前景绑定重建后，唯一观测ID才会进入coverage lineage。这样路径中间帧不会虚增科学覆盖率。

## 11. 最终模型完成

只有全部必需patch通过覆盖率、RMSE、法向一致性和点数门槛后，系统才执行：

1. 完整精扫来源链重放；
2. 多视图融合；
3. 正反面独立TSDF；
4. 网格生成；
5. 两侧和鳍片完整性检查；
6. 边界边、边界环和watertight检查；
7. 写入最终不可变重建；
8. 写入`FINE_COMPLETED`。

### 11.1 数学原理：先按标定位姿配准，再做有界同侧ICP

每个精扫点先用采集时绑定的相机位姿变换到base。所有视图共同做PCA确定主轴与法向，但正反侧
根据相机中心相对中面的符号永久分开：

$$
s_i=\operatorname{sign}\big((\mathbf c_i-\bar{\mathbf p})^\mathsf T\mathbf n\big)
\in\{-1,+1\}.
$$

残余ICP只在同一侧内执行，防止薄叶片正面吸附到几何相似的背面。对源点\(\mathbf p_i\)、对应
目标点\(\mathbf q_i\)和目标法向\(\mathbf n_i\)，点到面残差为：

$$
r_i=\mathbf n_i^\mathsf T(\mathbf q_i-(R\mathbf p_i+\mathbf t)).
$$

每轮求带鲁棒权重和位姿先验的最小二乘：

$$
\min_{\delta\boldsymbol\xi}
\sum_i w_i^2r_i^2+\lambda\|\delta\boldsymbol\xi\|_2^2.
$$

校正只有在对应点数足够、RMSE不恶化、平移校正和旋转校正都不超过配置上限时才接受；否则保留
原标定位姿。双侧中位高度差给出叶片实测厚度，若小到在体素尺度下不可分离则终止。

### 11.2 数学原理：正反面独立TSDF和薄壁保护

截断带不是无条件使用配置最大值。实际保护距离为：

$$
\mu=\min\!\left(\mu_{cfg},\ \beta\min(t_{blade},t_{fin,1},\ldots)\right),
$$

其中\(\beta\)是`thin_wall_band_fraction`。若\(\mu\)小于一个体素，重建直接失败，因为当前分辨率
无法表达该薄壁。

NumPy稀疏后备实现沿每条相机到表面射线，在表面前后\([-\mu,+\mu]\)采样。距表面有符号偏移
\(s\)的归一化值为：

$$
\phi=\operatorname{clip}(-s/\mu,-1,1),\qquad
\Phi(x)=\frac{\sum_k w_k\phi_k(x)}{\sum_k w_k}.
$$

Open3D可用且视图包含完整投影元数据时使用其可扩展TSDF；否则使用上述确定性稀疏实现。无论
后端哪一个，正反观测先进入两个独立volume，再分别提取\(\Phi=0\)等值面，避免一个面的自由空间
更新擦掉另一薄面。后备网格用marching tetrahedra在线性插值零交叉处生成三角形。

### 11.3 数学原理：“覆盖完成”和“最终完成”是两层门

第一层是固定schema-5全部必需patch的科学质量合取。它只触发最终重建。第二层重新回放完整精扫
lineage，并检查：

$$
G_{final}=G_{all\ patches}\land G_{views/side}\land G_{triangles/side}
\land G_{fin/front}\land G_{fin/back}\land G_{fin\ regions}
\land G_{topology}.
$$

网格边若只属于一个三角形就是边界边：

$$
E_b=\{e:\operatorname{triangleCount}(e)=1\}.
$$

边界环数是这些边组成的图的连通分量数。当前正式门槛要求边界边数=0、边界环数=0，且非空网格
满足watertight；同时正反侧都要有来源视图和三角形，每侧恰有一个鳍片且具双面证据，鳍片面、
鳍根、自由边区域全部完成。第一层通过而第二层失败时，正确终态仍是失败，不会写`fine_completed`。

成功标志：

- [ ] 控制台显示`Scan completed; no further motion is authorized.`
- [ ] 顶层链最后事件为`fine_completed`。
- [ ] 最终重建目录存在且可严格读取。
- [ ] GUI不再显示待批准运动。

实验模式的最终模型是可回放实验结果，但不会声称具有正式science acceptance。

## 12. 停止、急停和失败处理

### 正常主动停止

在控制台提示输入时键入`q`。等待程序确认stop并释放设备。不要直接关闭终端作为正常停止方式。

### 紧急情况

人员侵入、异常方向、碰撞风险、电缆拉扯或控制失常时，立即使用物理急停。软件`q`和Dashboard
stop不是安全等级急停替代品。急停后保存所有日志，不在原进程中继续。

### BLOCKED/FAILED

- [ ] 不删除失败目录。
- [ ] 不修改失败事件或资产。
- [ ] 保存控制台完整输出和`live_timeline`。
- [ ] 记录当时placement ID、run ID、最后动作和现场照片。
- [ ] 不移动实物时：新建`attempt-02`，沿用placement ID。
- [ ] 移动物体后：新建placement ID和attempt-01。

当前实验模式不允许`--resume`。失败后必须使用新run ID和新output；正式生产链完成验收后才使用
严格`--resume`。

### 数学原理：停止会切断授权链

运动许可不是状态无关的布尔值，而是与当前stop generation、路径、地图、策略和时限绑定的能力
凭证。若任一状态从\(X_k\)变成\(X_{k+1}\)，旧许可验证应为：

$$
\operatorname{valid}(T_k,X_{k+1})=0.
$$

正常`q`触发受控stop和资源释放；物理急停从独立安全链路切断运动能量，二者安全等级不同。
BLOCKED/FAILED表示系统无法证明某个必要命题，不代表已经证明有碰撞，但在fail-closed逻辑下：

$$
\text{UNKNOWN}\neq\text{SAFE},\qquad
\text{允许运动}\iff\text{所有必要证明均为CLEAR}.
$$

实验模式禁止resume，是因为当前运行链没有承诺可以从任意中断点完整恢复设备、地图、一次性permit
和in-flight segment的同一语义状态。新attempt保留原失败证据，同时从明确的初态重新建立链。

常见阻塞处理：

| 现象 | 正确处理 |
|---|---|
| `robot_mode=RUNNING` | 停止控制器程序，确认IDLE后开始全新attempt |
| hard ROI失败 | 重画多边形，先离线bootstrap-mask验证 |
| 三帧后仍MAPPING | 检查独立视角、有效深度、自遮罩和FREE票，不降低门槛 |
| FoundationStereo失败 | 保存attempt，检查CUDA/权重/显存，不覆盖输出 |
| token mismatch | 没有运动；重新粘贴当前控制台打印的完整token |
| occupancy/continuous proof BLOCKED | 不运动；检查真实障碍、UNKNOWN和工作空间，不临时豁免 |
| schema-5缺鳍片双面 | 允许规划器继续粗扫；候选耗尽则停止分析数据 |
| 精扫候选耗尽 | 停止，分析固定参考、掩模、IK/FK和质量证据 |
| 最终网格门槛失败 | 保留精扫数据，离线分析；不能把失败改写成完成 |

## 13. 运行后数据核对与拷贝

读取正式放置身份和终态：

```bash
uv run python -c "from biblade_fusion.storage.unknown_blade_experiment import read_unknown_blade_experiment; s=read_unknown_blade_experiment('data/experiments/blade-placement-20260831-01-attempt-01/experiment_handoff'); print(s.placement_id, s.experiment_id, s.latest_event.event_type, len(s.events))"
```

预期前两项分别为placement ID和run ID；成功运行的最后事件为`fine_completed`。

- [ ] 保留整个实验根，不只复制最后模型。
- [ ] 同时保留本次local配置、标定、接受资产、模型和代码提交号的记录。
- [ ] 进程完全退出后再复制数据。
- [ ] 使用校验和或保留文件时间/权限的复制方式。

实验资产内部绑定eiai绝对路径。复制到Vale后可以分析数组和事件，但严格语义回放仍应在原eiai
路径与原运行环境完成；不能修改metadata中的路径来伪造可移植性。

### 数学原理：为什么要复制整条链并校验

数组内容哈希不仅覆盖字节，还覆盖dtype与shape，可概括为：

$$
H_A=\operatorname{SHA256}(\operatorname{dtype}(A),\operatorname{shape}(A),\operatorname{bytes}(A)).
$$

元数据再引用这些数组哈希、父generation哈希和来源路径。只复制最终mesh会丢失“哪些帧、哪套
标定、哪条polygon/schema-5、哪些质量门产生了它”的证明。复制后逐文件校验和相等只能证明字节
未损坏；由于资产还绑定原eiai绝对路径，Vale上的普通分析与eiai原路径上的严格语义回放是两个
不同能力，不能通过手改metadata把前者伪装成后者。
