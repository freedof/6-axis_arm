# 验证指南

本文档集中记录项目验证命令、自动预检范围和用户验收规则。`AGENTS.md` 只保留入口索引；详细验证步骤放在这里。

## 验证原则

验证分两层：

```text
1. 自动预检：脚本、数值检查、GIF 文件检查通过。
2. 用户验收：用户逐个查看生成的 GIF，并明确确认通过。
```

不要因为脚本返回 `status: OK` 就把案例描述为最终通过。应描述为：

```text
automatic pre-check passed; waiting for user GIF confirmation
```

请求用户验收前，应说明：

```text
case id
case name
scene description
what should be visually confirmed
GIF path
automatic pre-check status
user acceptance status: pending
```

## 基础模型与运动学

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
.venv\Scripts\python src\sim\verify_kinematics.py
```

预期：

```text
FK matches MuJoCo tool0 pose to numerical precision.
Jacobian position rows match finite differences.
IK reaches target poses with micrometer-scale position error.
```

## 传统圆球往返 Demo

运行和渲染两目标往返：

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py --no-viewer
.venv\Scripts\python src\sim\render_roundtrip_gif.py
```

运行和渲染混合姿态往返：

```powershell
.venv\Scripts\python src\sim\demo_mixed_orientation_roundtrip.py --no-viewer
.venv\Scripts\python src\sim\render_mixed_orientation_gif.py
```

语义说明：

```text
Demo 1: red/blue targets use downward contact, tool0 local Z = world -Z.
Demo 2: red uses downward contact; blue uses horizontal contact, tool0 local Z = world +X.
```

目标坐标是 sphere center，不是 tool-center。脚本会将 tool0 目标偏移到与球相切：

```text
tool0_target = sphere_center + approach * (target_radius + tool_radius)
```

默认值：

```text
target_radius = 0.014 m
tool_radius   = 0.020 m
approach      = [0, 0, 1]
```

## RRT-Connect 与障碍规划

规划 smoke test：

```powershell
.venv\Scripts\python src\sim\verify_planning.py
.venv\Scripts\python src\sim\verify_planning_model.py
.venv\Scripts\python src\sim\verify_obstacle_planning.py
.venv\Scripts\python src\sim\verify_trajectory.py
```

渲染规划 GIF：

```powershell
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py --case-id TC-RRT-MULTI-005
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py --all
```

GIF 文件自动检查：

```powershell
.venv\Scripts\python src\sim\verify_render_gifs.py
```

RRT-Connect 案例、场景说明、GIF 路径和用户验收清单以此文档为准：

```text
docs/rrt_connect_test_report.md
```

规划验证中，可见目标球是 collision-enabled environment obstacles。末端目标仍与球相切，并保留小的数值接触容差。

## 夹爪与动态抓取

生成和验证夹爪模型：

```powershell
.venv\Scripts\python src\sim\gripper_model.py
.venv\Scripts\python src\sim\verify_gripper_model.py
```

生成和验证简化夹爪方块抓取场景：

```powershell
.venv\Scripts\python src\sim\gripper_pick_scene.py
.venv\Scripts\python src\sim\verify_gripper_pick.py
.venv\Scripts\python src\sim\verify_scene_separation.py
```

渲染抓取 GIF：

```powershell
.venv\Scripts\python src\sim\render_gripper_pick_gif.py
```

输出示例：

```text
outputs/gripper_pick/simplified_gripper_pick_cube.gif
```

抓取场景应与传统红/蓝圆球场景分离，不应包含 legacy roundtrip target spheres。

## D435i RGB-D 相机

生成和验证 D435i 场景：

```powershell
.venv\Scripts\python src\sim\d435i_model.py
.venv\Scripts\python src\sim\verify_d435i_camera.py
```

渲染 RGB/depth 预览：

```powershell
.venv\Scripts\python src\sim\render_d435i_preview.py
.venv\Scripts\python src\sim\render_d435i_preview.py --pose lift
.venv\Scripts\python src\sim\render_d435i_preview.py --pose scan
```

输出示例：

```text
outputs/d435i_preview/d435i_rgb.png
outputs/d435i_preview/d435i_noisy_depth_vis.png
```

## VL 识别与深度反投影

本地可重复验证：

```powershell
.venv\Scripts\python src\sim\verify_vl_region.py
```

真实 provider 验证。没有本地 key 时应返回 `status: SKIPPED`：

```powershell
.venv\Scripts\python src\sim\verify_openai_vl_provider.py
.venv\Scripts\python src\sim\verify_ark_coding_vl_provider.py
.venv\Scripts\python src\sim\verify_openrouter_vl_provider.py
```

OpenRouter / Gemini 当前推荐配置：

```json
{
  "providers": {
    "openrouter_vision": {
      "api_key": "YOUR_OPENROUTER_API_KEY",
      "base_url": "https://openrouter.ai/api/v1",
      "model": "google/gemini-3.5-flash"
    }
  }
}
```

真实 provider 配置文件：

```text
config/vl_providers.local.json
```

该文件包含 API Key，已被 `.gitignore` 忽略，不要提交。

## target_3d 到规划抓取

本地 color fixture 验证：

```powershell
.venv\Scripts\python src\sim\verify_vl_planned_pick.py --provider color_fixture
.venv\Scripts\python src\sim\verify_multi_view_vl_pick.py --provider color_fixture
```

OpenRouter 多视角验证示例：

```powershell
.venv\Scripts\python src\sim\verify_multi_view_vl_pick.py --provider openrouter_vision --model google/gemini-3.5-flash --output-dir outputs\end_to_end\openrouter_multi_view_vl_pick --gif outputs\end_to_end\openrouter_multi_view_vl_pick\openrouter_multi_view_vl_pick_cube.gif --camera-width 424 --camera-height 240 --max-parallel-vl 4 --frames 360 --fps 20
```

当前外部真实 VL provider 至少需要 `2` 个视角通过几何和一致性检查，才会进入抓取规划。

## MCP Server

运行本地 MCP server：

```powershell
.venv\Scripts\python src\mcp_robot\server.py
```

验证 MCP server：

```powershell
.venv\Scripts\python src\mcp_robot\verify_server.py
```

MCP 工具说明以此文档为准：

```text
docs/mcp_robot_server.md
```

## 完整回归建议

在模型、运动学、规划、抓取、D435i 或 VL 链路发生较大变更后，建议运行：

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
.venv\Scripts\python src\sim\verify_kinematics.py
.venv\Scripts\python src\sim\verify_planning.py
.venv\Scripts\python src\sim\verify_planning_model.py
.venv\Scripts\python src\sim\verify_obstacle_planning.py
.venv\Scripts\python src\sim\verify_trajectory.py
.venv\Scripts\python src\sim\verify_gripper_model.py
.venv\Scripts\python src\sim\verify_gripper_pick.py
.venv\Scripts\python src\sim\verify_scene_separation.py
.venv\Scripts\python src\sim\verify_d435i_camera.py
.venv\Scripts\python src\sim\verify_vl_region.py
.venv\Scripts\python src\mcp_robot\verify_server.py
```

涉及渲染或用户验收时，还需要刷新相应 GIF。每个 GIF 的场景描述和验收标准应引用对应测试文档。
