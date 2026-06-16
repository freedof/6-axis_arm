# CR5 机器人 MCP Server

## 目标

本 MCP server 把当前项目中已经验证过的机械臂能力封装成大模型可调用的工具。

第一版重点不是开放底层关节控制，而是开放稳定的高层技能：

```text
查询能力
查询场景
生成夹爪模型
生成抓取场景
生成 D435i 相机场景
渲染 D435i RGB-D 预览
仿真抓取方块
渲染抓取 GIF
执行 pick_cube 技能
```

协议入口：

```powershell
.venv\Scripts\python src\mcp_robot\server.py
```

验证入口：

```powershell
.venv\Scripts\python src\mcp_robot\verify_server.py
```

## 工具列表

### get_robot_capabilities

返回当前机器人能力：

```text
CR5 简化模型
MuJoCo 仿真
6 轴基准模型
8 执行器夹爪模型
可执行的抓取方块技能
自动预检 + GIF 人工验收流程
```

### list_available_scenes

列出当前 MCP 知道的场景：

```text
cr5_simplified
cr5_with_gripper
gripper_pick_cube
gripper_pick_cube_d435i
```

### get_scene_state

返回场景的结构化信息。当前 `gripper_pick_cube` 会返回：

```text
红色动态方块 grasp_cube
桌面 pick_table
方块尺寸、质量、摩擦参数
模型路径
```

### generate_gripper_model

生成：

```text
assets/dobot_cr5/mjcf/cr5_with_gripper.xml
```

### generate_pick_scene

生成：

```text
assets/dobot_cr5/mjcf/cr5_gripper_pick_scene.xml
```

### simulate_pick_cube

只运行动力学抓取仿真，不渲染 GIF。

返回示例：

```json
{
  "status": "automatic_precheck_passed",
  "scene_id": "gripper_pick_cube",
  "object": "grasp_cube",
  "metrics": {
    "initial_cube_pos_m": [0.35, -0.55, 0.065],
    "final_cube_pos_m": [0.34748, -0.54025, 0.14736],
    "max_cube_z_m": 0.14844,
    "lifted": true
  },
  "user_acceptance": "pending"
}
```

### render_pick_cube_gif

渲染并验证抓取 GIF。

默认输出：

```text
outputs/gripper_pick/simplified_gripper_pick_cube.gif
```

### pick_cube

执行完整抓取技能：

```text
生成场景
运行动力学仿真
检查方块是否被抬起
可选渲染 GIF
返回结构化结果
```

典型 MCP 调用参数：

```json
{
  "render_gif": true,
  "frames": 160,
  "fps": 20,
  "width": 960,
  "height": 720,
  "show_sites": false
}
```

## 验收原则

MCP 工具返回 `automatic_precheck_passed` 只表示自动检查通过。

最终是否通过仍然需要用户查看 GIF 并确认：

```text
automatic pre-check passed; waiting for user GIF confirmation
```

## 设计约束

当前 MCP server 遵守这些约束：

```text
1. 大模型不直接控制关节角。
2. 大模型调用高层技能。
3. 每个重要技能返回结构化指标。
4. 可视化任务必须生成 GIF。
5. 用户保留最终验收权。
```

## 后续扩展

建议按以下顺序继续扩展：

```text
1. place_cube
2. pick_and_place_cube
3. plan_to_pose
4. plan_pick_path
5. 多物体场景状态读取
6. RGB-D 相机状态导出
7. 大模型技能编排器
```
## D435i 相机工具

当前 MCP server 也暴露了夹爪根部 D435i 的基础感知能力。

### generate_d435i_scene

生成带 D435i 的夹爪抓取场景：

```text
assets/dobot_cr5/mjcf/cr5_gripper_d435i_pick_scene.xml
```

返回：

```json
{
  "status": "ok",
  "scene_id": "gripper_pick_cube_d435i",
  "camera_names": ["d435i_depth", "d435i_rgb"]
}
```

### render_d435i_preview

从夹爪根部 D435i 渲染一次 RGB-D 观测。

典型 MCP 调用参数：

```json
{
  "width": 424,
  "height": 240,
  "pose": "above",
  "output_dir": "outputs/d435i_preview"
}
```

输出内容包括：

```text
RGB 图片
外部对照 RGB 图片
raw depth .npy
noisy depth .npy
raw depth 可视化 PNG
noisy depth 可视化 PNG
intrinsics
world-to-camera extrinsic
raw/noisy depth 有效像素统计
```

默认使用 `above` 预抓取观察位，因为 `grasp` 贴近姿态可能低于 D435i
约 0.17 m 的最小有效深度。
