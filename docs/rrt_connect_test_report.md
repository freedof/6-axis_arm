# RRT-Connect 测试用例报告

## 1. 测试目标

本文档说明当前 Dobot CR5 自研运动规划栈的验证算例、自动预检命令、
GIF 输出以及人工验收要求。

当前验证覆盖：

```text
模型加载
FK / Jacobian / IK
planning collision model
多目标点 RRT-Connect
多末端接触姿态 RRT-Connect
障碍物绕行
Jacobian 奇异性裕度
路径 shortcut
关节轨迹时间参数化
GIF 渲染验证
```

重要验收原则：

```text
自动脚本通过只代表 automatic pre-check passed。
最终通过必须由用户查看对应 GIF 后确认。
```

## 2. 测试环境

工作目录：

```text
F:\6-axis arm
```

Python 解释器：

```powershell
.venv\Scripts\python
```

模型：

```text
assets/dobot_cr5/mjcf/cr5_simplified.xml
assets/dobot_cr5/mjcf/cr5_planning.xml
```

说明：

```text
cr5_simplified.xml 用于显示、FK/MuJoCo 对比和基础 demo。
cr5_planning.xml 用于规划碰撞检查。
```

`cr5_planning.xml` 由 `src/sim/planning_model.py` 生成：

```text
visual geoms       collision-disabled
collision_* geoms  collision-enabled
target_sphere      collision-enabled
target_sphere_b    collision-enabled
floor_collision    collision-enabled
```

`floor_collision` 当前位于 `z = 0 m`，与视觉地面平面对齐，并作为当前
规划验证中的基平面/地面障碍。后续获得更精确的机器人 collision geometry
后，需要继续校准桌面、夹具或真实基平面高度。

红/蓝目标小球本身也作为规划障碍参与碰撞检查。端点目标仍是工具与小球
表面相切；碰撞检查允许 `1e-6 m` 级别的数值容差，用于避免相切状态被
MuJoCo 的浮点误差误报为穿透。

## 3. 自动预检命令

基础模型与运动学：

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
.venv\Scripts\python src\sim\verify_kinematics.py
```

规划碰撞模型：

```powershell
.venv\Scripts\python src\sim\verify_planning_model.py
```

RRT 多 case 验证：

```powershell
.venv\Scripts\python src\sim\verify_planning.py
```

奇异性裕度在 RRT 多 case、障碍绕行和轨迹时间参数化验证中同步检查。
当前阈值定义在：

```text
src/planning/singularity.py
```

当前阈值：

```text
max_condition_number = 1000
min_singular_value   = 1e-3
min_manipulability   = 5e-4
```

障碍物绕行验证：

```powershell
.venv\Scripts\python src\sim\verify_obstacle_planning.py
```

轨迹时间参数化验证：

```powershell
.venv\Scripts\python src\sim\verify_trajectory.py
```

GIF 文件结构验证：

```powershell
.venv\Scripts\python src\sim\verify_render_gifs.py
```

## 4. Planning Collision Model 验证

测试编号：

```text
TC-MODEL-PLANNING-001
```

执行命令：

```powershell
.venv\Scripts\python src\sim\verify_planning_model.py
```

验证内容：

```text
1. 生成 assets/dobot_cr5/mjcf/cr5_planning.xml。
2. 检查 floor_collision 已启用碰撞。
3. 检查 target_sphere / target_sphere_b 已启用碰撞。
4. 检查 collision_base / collision_link* / collision_tool 已启用碰撞。
5. 检查 visual geoms 已关闭碰撞。
```

通过标准：

```text
floor_collision contype/conaffinity 均非 0。
target_sphere / target_sphere_b contype/conaffinity 均非 0。
collision_* geoms contype/conaffinity 均非 0。
visual geoms contype/conaffinity 均为 0。
```

该测试没有单独 GIF；它是模型结构预检。最终验收仍以各规划 case GIF 为准。

## 5. 多目标点与多姿态 RRT 算例

所有 case 定义在：

```text
src/sim/planning_cases.py
```

每个 case 的自动预检流程：

```text
1. 根据球心和接触姿态计算 tool0 目标位姿。
2. 求解 A 点 IK。
3. 使用 A 点解作为优先 seed 求解 B 点 IK。
4. 检查 A/B 端点关节限位和碰撞。
5. 强制执行 RRT-Connect，try_direct=False。
6. 检查 RRT 路径端点正确。
7. 对 RRT 路径逐段采样并检查碰撞和 Jacobian 奇异性裕度。
8. 执行 shortcut。
9. 对 shortcut 后路径逐段采样并检查碰撞和 Jacobian 奇异性裕度。
```

统一通过标准：

```text
IK 成功。
A/B 端点有效。
目标小球作为障碍参与检查，端点只允许相切数值容差内的接触。
RRT-Connect 成功。
RRT 路径逐段有效。
shortcut 后路径逐段有效。
路径采样点满足 Jacobian 奇异性裕度。
对应 GIF 已生成。
用户视觉确认 GIF 后，case 才算最终通过。
```

### TC-RRT-MULTI-001：近距离竖直接触

场景：

```text
A: 球心 [0.35, -0.55, 0.20]，tool0 local Z = world -Z
B: 球心 [0.31, -0.53, 0.24]，tool0 local Z = world -Z
```

目的：

```text
验证短距离、同姿态、竖直接触目标之间的规划链路。
```

GIF：

```text
outputs/test_gifs/tc_rrt_multi_001_planned_test.gif
```

人工确认重点：

```text
机械臂应在两个相近竖直接触目标之间平稳往返。
路径不应出现明显跳变、穿模或异常绕远。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-MULTI-002：远距离竖直接触

场景：

```text
A: 球心 [0.42, -0.50, 0.18]，tool0 local Z = world -Z
B: 球心 [0.15, -0.63, 0.34]，tool0 local Z = world -Z
```

目的：

```text
验证较大工作空间跨度下的同姿态竖直接触规划。
```

GIF：

```text
outputs/test_gifs/tc_rrt_multi_002_planned_test.gif
```

人工确认重点：

```text
机械臂应完成较大范围移动，并保持两端竖直接触姿态。
运动不应出现关节突跳或明显不自然的大幅绕行。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-MULTI-003：竖直高度变化

场景：

```text
A: 球心 [0.31, -0.56, 0.20]，tool0 local Z = world -Z
B: 球心 [0.31, -0.56, 0.38]，tool0 local Z = world -Z
```

目的：

```text
验证 XY 接近但 Z 高度差明显时，肘部和腕部配合是否稳定。
```

GIF：

```text
outputs/test_gifs/tc_rrt_multi_003_planned_test.gif
```

人工确认重点：

```text
机械臂应主要完成高度方向变化，末端保持竖直接触方向。
不应出现向侧面大幅偏离或明显构型跳变。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-MULTI-004：竖直到水平 +X

场景：

```text
A: 球心 [0.35, -0.55, 0.20]，tool0 local Z = world -Z
B: 球心 [0.20, -0.60, 0.30]，tool0 local Z = world +X
```

目的：

```text
验证末端从竖直接触切换到水平 +X 接触时的位置和姿态规划。
```

GIF：

```text
outputs/test_gifs/tc_rrt_multi_004_planned_test.gif
```

人工确认重点：

```text
A 点应为向下接触，B 点应为水平 +X 接触。
过渡过程中姿态变化应连续，不应出现腕部异常翻转。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-MULTI-005：左右跨越水平接触

场景：

```text
A: 球心 [0.26, -0.55, 0.42]，tool0 local Z = world +X
B: 球心 [0.26,  0.42, 0.42]，tool0 local Z = world -X
```

目的：

```text
验证 base 关节大角度旋转和左右两侧水平接触姿态。
```

GIF：

```text
outputs/test_gifs/tc_rrt_multi_005_planned_test.gif
```

人工确认重点：

```text
机械臂应完成左右跨越运动。
两端均应呈水平接触姿态。
运动不应出现明显穿过自身或环境障碍的现象。
该 case 使用优先 IK seed 选择远离奇异的 IK 分支，避免左右跨越路径穿过腕部/肘部奇异区域。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-MULTI-006：接近工作空间边界

场景：

```text
A: 球心 [0.52, -0.36, 0.22]，tool0 local Z = world -Z
B: 球心 [0.46, -0.16, 0.35]，tool0 local Z = world +Y
```

目的：

```text
验证接近可达边缘但仍可达目标下的 IK 和规划稳定性。
```

GIF：

```text
outputs/test_gifs/tc_rrt_multi_006_planned_test.gif
```

人工确认重点：

```text
机械臂应接近工作空间边缘但保持稳定运动。
不应出现明显关节限位附近的抖动、跳变或不可解释绕行。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

## 6. 障碍物绕行算例组

所有障碍 case 定义在：

```text
src/sim/obstacle_scene.py
```

执行命令：

```powershell
.venv\Scripts\python src\sim\verify_obstacle_planning.py
```

生成正式人工验收 GIF：

```powershell
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py --all
```

说明：

```text
障碍物人工验收 GIF 默认渲染 shortcut 后的路径。
raw RRT 路径仍会在自动预检中验证，但它只用于调试规划树，不作为视觉验收 GIF。
如果需要查看 raw RRT，可使用 --raw-rrt。
渲染到达 A/B 目标点时默认各停留约 1 s，可用 --target-dwell-seconds 调整。
```

每个障碍 case 的自动预检流程：

```text
1. 基于 cr5_planning.xml 生成带障碍物的 MJCF。
2. 求解对应基础运动 case 的 A/B 端点 IK。
3. 验证 A/B 端点本身有效，不能一开始就碰撞。
4. 验证 A 到 B 的直接关节空间路径会碰撞。
5. 运行 RRT-Connect 寻找绕行路径。
6. 验证 RRT 路径逐段无碰撞，并满足 Jacobian 奇异性裕度。
7. 执行 shortcut 并验证路径仍无碰撞、仍满足奇异性裕度。
8. 执行时间参数化并验证采样状态有效，且不靠近奇异。
9. 渲染测试 GIF，并用 Pillow 做文件结构预检。
```

统一通过标准：

```text
A/B 端点有效。
直接 A->B 路径被障碍挡住。
RRT-Connect 成功。
RRT 路径逐段有效。
shortcut 后路径逐段有效。
时间参数化轨迹采样有效。
路径和轨迹采样点满足 Jacobian 奇异性裕度。
对应 GIF 已生成。
用户视觉确认 GIF 后，case 才算最终通过。
```

### TC-RRT-OBSTACLE-001：单箱体阻挡近距离竖直接触

基础 case：

```text
TC-RRT-MULTI-001：近距离竖直接触
```

场景：

```text
一个中等箱体放在短距离竖直接触运动的中间扫掠区域。
直接 A->B 关节空间路径会被箱体挡住。
```

障碍物：

```text
类型：box
位置：[0.24977, -0.62073, 0.17463]
半尺寸：[0.07, 0.07, 0.14]
```

GIF：

```text
outputs/test_gifs/tc_rrt_obstacle_001_test.gif
outputs/obstacle_scene/tc_rrt_obstacle_001.gif
```

人工确认重点：

```text
机械臂应绕开单个箱体。
绕行过程中机械臂和箱体之间不应出现明显穿模。
动作应连续，不应出现异常瞬移。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-OBSTACLE-002：长距离竖直接触中的高箱体绕行

基础 case：

```text
TC-RRT-MULTI-002：远距离竖直接触
```

场景：

```text
两个较远的竖直接触目标之间放置一个较高箱体。
该 case 用来验证较大工作空间跨度下的障碍绕行。
```

障碍物：

```text
类型：box
位置：[0.285, -0.575, 0.245]
半尺寸：[0.08, 0.075, 0.18]
```

GIF：

```text
outputs/test_gifs/tc_rrt_obstacle_002_test.gif
outputs/obstacle_scene/tc_rrt_obstacle_002.gif
```

人工确认重点：

```text
机械臂应完成较大范围绕行。
末端两端仍应保持竖直接触姿态。
路径不应为了绕障出现明显不合理的大幅乱绕。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-OBSTACLE-003：左右跨越中的扫掠区障碍

基础 case：

```text
TC-RRT-MULTI-005：左右跨越水平接触
```

场景：

```text
障碍物不只看球心连线，而是放在机械臂左右跨越时实际扫过的区域。
该 case 用来验证整条机械臂的绕障，而不是只验证末端点绕障。
```

障碍物：

```text
类型：box
位置：[0.060, -0.200, 0.200]
半尺寸：[0.05, 0.05, 0.10]
```

GIF：

```text
outputs/test_gifs/tc_rrt_obstacle_003_test.gif
outputs/obstacle_scene/tc_rrt_obstacle_003.gif
```

人工确认重点：

```text
机械臂左右跨越时不应扫过障碍箱体。
需要重点观察大臂、肘部和腕部是否与障碍穿模。
路径可以比无障碍时更绕，但不应出现构型跳变。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

### TC-RRT-OBSTACLE-004：双箱体窄通道绕行

基础 case：

```text
TC-RRT-MULTI-004：竖直到水平 +X
```

场景：

```text
两个小箱体同时存在，形成局部受限空间。
该 case 用来验证多障碍物和姿态切换同时存在时的绕行。
```

障碍物：

```text
box 1 位置：[0.160, -0.580, 0.160]，半尺寸：[0.04, 0.04, 0.10]
box 2 位置：[0.280, -0.580, 0.100]，半尺寸：[0.04, 0.04, 0.10]
```

GIF：

```text
outputs/test_gifs/tc_rrt_obstacle_004_test.gif
outputs/obstacle_scene/tc_rrt_obstacle_004.gif
```

人工确认重点：

```text
机械臂应避开两个箱体，而不是只避开其中一个。
末端应从竖直接触连续切换到水平 +X 接触。
双障碍附近不应出现明显穿模、瞬移或腕部异常翻转。
```

验收状态：

```text
automatic pre-check: required
user acceptance: pending
```

## 7. 轨迹时间参数化验证

执行命令：

```powershell
.venv\Scripts\python src\sim\verify_trajectory.py
```

测试目标：

```text
验证 raw RRT path 和 shortcut path 都可以转换为带时间戳的平滑关节轨迹，
并且采样后的轨迹满足速度、加速度、碰撞和 Jacobian 奇异性裕度约束。
```

轨迹参数：

```text
max_joint_velocity = 0.8 rad/s
max_joint_acceleration = 1.6 rad/s^2
sample_dt = 0.02 s
```

通过标准：

```text
所有 TC-RRT-MULTI case 的 raw trajectory 有效。
所有 TC-RRT-MULTI case 的 shortcut trajectory 有效。
采样速度不超过限制。
采样加速度不超过限制。
所有采样状态通过碰撞检查。
所有采样状态通过 Jacobian 奇异性裕度检查。
```

该测试没有单独 GIF；它用于验证规划 GIF/live viewer 所用的时间参数化轨迹。

## 8. GIF 文件结构验证

执行命令：

```powershell
.venv\Scripts\python src\sim\verify_render_gifs.py
```

测试 GIF 参数：

```text
width = 960 px
height = 720 px
frames = 48
fps = 12
duration = 4 s
target_dwell_seconds = 1.0 s
shortcut = False
```

Pillow 预检内容：

```text
文件存在。
文件大小大于 0。
文件格式为 GIF。
GIF 尺寸正确。
GIF 存储帧数不超过预期，且总播放时长接近预期。
渲染时间轴在 A/B 目标点各包含约 1 s 停留。
第一帧不是空白画面。
第一帧和中间帧存在可见差异。
```

注意：

```text
Pillow 预检只说明 GIF 文件结构和基本画面变化有效。
是否符合规划预期，必须由用户查看 GIF 后确认。
```

## 9. 验收汇报格式

每次请求用户验收时，应按以下格式列出：

```text
case id:
case name:
scene description:
what to visually confirm:
gif path:
automatic pre-check status:
user acceptance status: pending
```

禁止将脚本 `status: OK` 直接描述为最终通过。应描述为：

```text
automatic pre-check passed; waiting for user GIF confirmation
```

## 10. 当前尚未覆盖

```text
非 box 障碍形状
更多障碍尺寸组合
更严格窄通道和低净空规划
无可行路径失败用例
带安全距离膨胀的碰撞检查
轨迹导出格式
更高阶连续性约束
```
