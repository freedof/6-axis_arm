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

规划检查中会忽略旧 roundtrip 演示用的 `target_sphere` 和 `target_sphere_b`，否则它们会作为无关调试障碍物干扰夹爪末端。

## MCP 工具

新增两个 MCP 工具：

```text
plan_pick_from_target_3d
vl_pick_cube
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
