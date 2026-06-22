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
通过 VL 风格区域定位目标
将目标区域结合 depth 反投影到 3D
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
gripper_multi_object
gripper_multi_object_d435i
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


### generate_multi_object_scene

生成多物体桌面场景，用于语言指令目标选择、VL 识别和后续抓取验证。默认包含不同颜色的方块、圆柱体和桌面托盘，并可同时生成 D435i 相机场景。

典型 MCP 调用参数：

```json
{
  "include_d435i": true
}
```

### parse_language_goal

把中文或英文桌面操作指令解析为结构化目标：动作、目标颜色、目标形状、候选物体、放置区域和供 VL 使用的目标提示。

典型 MCP 调用参数：

```json
{
  "instruction": "把所有圆柱体夹到托盘中"
}
```

返回结果中，`status=ok` 表示目标唯一；`status=ambiguous` 表示需要用户或上层策略继续消歧；`status=no_match` 表示当前场景没有匹配物体。


### multi_object_vl_locate

先解析语言指令，再在多物体 D435i 场景中执行多视角 VL 定位和深度反投影，返回所选物体的融合 `target_3d`。

典型 MCP 调用参数：

```json
{
  "instruction": "夹取蓝色方块",
  "provider": "color_fixture",
  "poses": ["scan_high", "scan_front_high", "scan_left_high", "scan_right_high"]
}
```

### language_multi_view_pick_and_place

执行完整闭环：语言目标解析、多物体多视角 VL 识别、深度定位、RRT-Connect pick-and-place 规划、MuJoCo 动力学仿真和可选 GIF 渲染。

默认验证指令：

```text
把蓝色方块放到托盘中
```

验证 GIF 示例：

```text
outputs/pick_place/multi_object_vl_pick_place/blue_cube_tray_pick_place.gif
```


集合指令示例：

```text
把所有圆柱体夹到托盘中
```

当前集合模式会将指令拆成多个单目标子任务，并为每个子任务生成一个 GIF，例如：

```text
outputs/pick_place/collect_cylinders_to_tray/gifs/01_green_cylinder.gif
outputs/pick_place/collect_cylinders_to_tray/gifs/02_yellow_cylinder.gif
```
该工具返回 `automatic_precheck_passed` 仍只表示自动预检通过，最终是否接受需要用户查看 GIF 后确认。
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

视觉定位建议使用 `scan` 姿态：

```json
{
  "width": 424,
  "height": 240,
  "pose": "scan",
  "output_dir": "outputs/d435i_preview"
}
```

## VL 区域定位工具

当前 MCP server 增加了三类感知工具：

```text
vl_locate_object_region
estimate_region_3d
vl_locate_object_3d
```

### vl_locate_object_region

输入 RGB 图像和文本目标意图，返回 VL 风格的 2D 区域。

当前支持六种 provider：

```text
color_fixture
manual_region
codex_vision
openai_vision
ark_coding_vision
openrouter_vision
```

`color_fixture` 用于本地可重复测试；`manual_region` 用于调用方手动传入 bbox/point；`codex_vision` 用于由当前 Codex 会话看图后传入 bbox/point，并在结果中明确记录为 Codex-in-the-loop VL；`openai_vision` 会调用 OpenAI Responses API；`ark_coding_vision` 会调用火山方舟 coding plan 的 OpenAI-compatible chat-completions endpoint；`openrouter_vision` 会调用 OpenRouter 的 OpenAI-compatible chat-completions endpoint，可用于 Gemini 等视觉模型。真实 provider 的 API Key、Base URL、模型名从 `config/vl_providers.local.json` 读取。

首次配置：

```powershell
Copy-Item config\vl_providers.example.json config\vl_providers.local.json
```

然后把 `config/vl_providers.local.json` 中的 `api_key` 改成自己的 key。该本地文件已被 `.gitignore` 忽略。

典型调用：

```json
{
  "prompt": "pick the red block",
  "provider": "color_fixture",
  "pose": "scan",
  "width": 424,
  "height": 240
}
```

真实 VL 调用示例：

```json
{
  "prompt": "pick the red block",
  "provider": "openai_vision",
  "model": "gpt-5.5",
  "pose": "scan",
  "width": 424,
  "height": 240
}
```

火山方舟 coding plan 调用示例：

```json
{
  "prompt": "pick the red block",
  "provider": "ark_coding_vision",
  "model": "glm-5.2",
  "pose": "scan",
  "width": 424,
  "height": 240
}
```

OpenRouter / Gemini 视觉模型调用示例：
```json
{
  "prompt": "Pick the small red cube block on the tabletop.",
  "provider": "openrouter_vision",
  "model": "google/gemini-3.5-flash",
  "pose": "scan_high",
  "width": 424,
  "height": 240
}
```

对应本地配置文件：

```text
config/vl_providers.local.json
```

参考模板：

```text
config/vl_providers.example.json
```

手动区域调试示例：

```json
{
  "prompt": "pick the red block",
  "provider": "manual_region",
  "manual_region": {
    "type": "bbox",
    "label": "red block",
    "bbox_xyxy": [199, 153, 225, 181],
    "confidence": 1.0
  }
}
```

返回重点：

```text
bbox_xyxy
confidence
overlay_path
RGB-D observation files
```

### estimate_region_3d

输入 2D 区域、depth 文件、相机内参和 world-to-camera 外参，返回世界坐标下的 3D 目标点。

该工具只负责几何反投影，不负责选择目标。

### vl_locate_object_3d

组合工具，先定位 2D 区域，再用 depth 得到 3D 目标点。

典型结果：

```json
{
  "target_3d": {
    "center_world_m": [0.349362, -0.549244, 0.095],
    "depth_m": 0.239994,
    "valid_pixel_count": 728
  }
}
```

这里得到的是可见表面附近的目标点，可作为后续生成抓取姿态的输入，不等价于最终抓取动作。
