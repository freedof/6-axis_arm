# VL 到 RRT-Connect 抓取闭环说明

## 目标

本阶段把 `target_3d` 真正接入抓取链路，不再只是输出一个 3D 点。当前闭环是：

```text
用户指令
  -> D435i RGB-D 扫描
  -> VL provider 输出目标 bbox
  -> 深度图在 bbox 内反投影得到 target_3d
  -> 根据 target_3d 生成抓取姿态
  -> RRT-Connect 规划 ready/above/grasp/lift 三段关节路径
  -> MuJoCo 动力学仿真夹取
  -> 渲染 GIF，等待用户验收
```

## target_3d 的含义

`target_3d.center_world_m` 目前表示目标在相机中可见的表面位置。对桌面红色方块来说，这个点接近方块上表面，而不是方块几何中心。

因此抓取前会做一次转换：

```text
cube_center = target_3d.center_world_m - [0, 0, cube_half_height]
```

当前方块半高来自场景定义：

```text
src/sim/gripper_pick_scene.py
CUBE_HALF_SIZE[2] = 0.030 m
```

## 抓取姿态生成

由估计出的 `cube_center` 生成三个关键抓取姿态：

```text
above: 位于方块中心上方 0.120 m
grasp: 夹爪中心对准方块中心
lift:  位于方块中心上方 0.150 m
```

这些姿态仍然通过现有 IK 求解，夹爪采用当前简化平行夹爪。姿态约束为自上而下抓取。

## RRT-Connect 规划

抓取动作不是直接关节插值，而是分成三段做 RRT-Connect：

```text
ready_to_above
above_to_grasp
grasp_to_lift
```

每段都会经过：

```text
关节限位检查
MuJoCo-backed 碰撞检查
Jacobian 奇异性裕度检查
RRT-Connect 搜索
shortcut 平滑
时间参数化
```

抓取规划使用独立的 pick planning 场景：

```text
assets/dobot_cr5/mjcf/cr5_gripper_pick_planning.xml
```

该场景从方块抓取场景派生，不包含旧 roundtrip 演示用的红蓝小球
`target_sphere` / `target_sphere_b`。红蓝小球只保留在 roundtrip/RRT 验证
场景中：

```text
assets/dobot_cr5/mjcf/cr5_simplified.xml
assets/dobot_cr5/mjcf/cr5_planning.xml
```

## MCP 工具

新增两个 MCP 工具：

## 执行速度和超调控制

抓取执行按动作阶段分别限速，而不是使用一个全局慢速参数：

```text
ready_to_above:  max_joint_velocity = 0.75 rad/s, max_joint_acceleration = 1.40 rad/s^2
above_to_grasp:  max_joint_velocity = 0.24 rad/s, max_joint_acceleration = 0.45 rad/s^2
grasp_to_lift:   max_joint_velocity = 0.40 rad/s, max_joint_acceleration = 0.70 rad/s^2
above dwell:     0.4 s
```

其中 `ready_to_above` 保持接近之前版本的起步速度，避免一开始运动到最高点前显得过慢；
`above_to_grasp` 是从高位下降到抓取点的关键段，单独放慢并在高位短暂停留，用来减少接近
目标时的“冲”和“过头”感觉。

MuJoCo 仿真和 GIF 渲染时，关节目标会在每个 physics step 按连续时间重新采样，
而不是只按 GIF 帧率更新。这样可以避免 20Hz 帧率目标造成阶跃追踪，减少机械臂
看起来“冲得很快”或“过头”的现象。

抓取场景中的关节 position actuator 也使用较保守的伺服参数：

```text
joint kp         = 420
joint kv         = 60
joint forcerange = -320 320
```

这些参数优先服务于稳定、可观察的仿真抓取，而不是追求最快动作。

```text
plan_pick_from_target_3d
vl_pick_cube
multi_view_vl_locate_object_3d
multi_view_vl_pick_cube
```

`plan_pick_from_target_3d` 用于已经拿到 `target_3d` 的情况：

```json
{
  "target_3d": {
    "center_world_m": [0.349, -0.549, 0.095]
  },
  "render_gif": true
}
```

`vl_pick_cube` 用于完整闭环：

```json
{
  "prompt": "pick the red block",
  "provider": "color_fixture",
  "pose": "scan",
  "render_gif": true
}
```

`multi_view_vl_pick_cube` 用于更稳的多视角闭环：

```json
{
  "prompt": "pick the red block",
  "provider": "ark_coding_vision",
  "model": "doubao-seed-2.0-pro",
  "poses": ["scan", "scan_left", "scan_right", "scan_high"],
  "max_parallel_vl": 4,
  "render_gif": true
}
```

多视角流程会先顺序生成多个 D435i 视角，再并行调用 VL provider。这样远端
VL 调用不会简单变成 4 倍等待时间。

当前多视角扫描位姿采用更高的拍照位置：`scan`、`scan_left`、`scan_right`
等视角相对方块中心约高 `0.30 m`，`scan_high` 约高 `0.38 m`。这样可以减少
低视角下桌面透视、阴影和方块侧面投影导致的 bbox 偏大问题。

每个视角都会输出：

```text
RGB 图
VL overlay
bbox
target_3d
accepted / rejected
reject_reason
```

融合时会先拒绝桌面高度附近的候选，再做 3D 空间聚类，只融合属于同一目标簇
的候选。这个设计是为了支持后续多个同形状方块：VL 需要根据用户指令中的颜色、
位置或任务语义选择目标；如果不同视角选到了不同方块，3D 聚类会暴露不一致，
而不是把不同方块的位置平均掉。

真实 VL provider 可改为：

```json
{
  "prompt": "pick the red block",
  "provider": "ark_coding_vision",
  "model": "doubao-seed-2.0-pro",
  "pose": "scan",
  "render_gif": true
}
```

## 自动验证

离线可重复验证：

```powershell
.venv\Scripts\python src\sim\verify_vl_planned_pick.py --provider color_fixture
.venv\Scripts\python src\sim\verify_multi_view_vl_pick.py --provider color_fixture
```

使用本地配置的 Ark coding plan 验证：

```powershell
.venv\Scripts\python src\sim\verify_vl_planned_pick.py --provider ark_coding_vision --model doubao-seed-2.0-pro
```

这一步是可选真实 provider 验证。当前已验证 `doubao-seed-2.0-pro` 可以接收
图片并完成闭环。如果服务返回 `Model only support text input`，说明当前选择的
coding 模型不能接收图片，不能作为本项目的 VL provider；此时需要换成支持图像
输入的模型或 endpoint。

验证脚本会输出：

```text
RGB 图路径
VL overlay 路径
bbox
target_3d
估计出的 cube_center
三段 RRT-Connect 规划摘要
GIF 路径
抓取仿真 lift 指标
```

## 验收规则

脚本通过只代表自动预检通过。生成的 GIF 仍需要用户最终确认：

```text
automatic pre-check passed; waiting for user GIF confirmation
```
