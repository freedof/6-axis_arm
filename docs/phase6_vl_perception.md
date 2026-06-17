# Phase 6 VL 视觉定位接入说明

## 当前目标

本阶段先把“视觉语言模型选择目标区域”接入到机械臂感知链路中。完整链路是：

```text
D435i RGB 图像
  -> VL 模型根据文本指令输出目标区域
  -> D435i depth 在该区域内取深度
  -> 通过相机内外参反投影到世界坐标
  -> 后续生成抓取姿态和抓取路径
```

当前已经实现的是前半段：从一张 D435i RGB-D 观测中，得到目标物体的 2D 区域和 3D 位置估计。它还不是完整的自动抓取闭环。

## 重要说明

当前仓库支持四种 provider：

```text
color_fixture   本地颜色规则，用于可重复自动测试
manual_region   调用方手动传 bbox/point，用于调试和验收复现
openai_vision   调用 OpenAI Responses API 的真实视觉语言模型
ark_coding_vision  调用火山方舟 coding plan 的 OpenAI-compatible 接口
```

代码入口：

```text
src/perception/vl_region.py
```

真实 provider 的配置方式：

```powershell
Copy-Item config\vl_providers.example.json config\vl_providers.local.json
```

然后编辑：

```text
config/vl_providers.local.json
```

`vl_providers.local.json` 包含 API Key，已被 `.gitignore` 忽略，不要提交到仓库。

`color_fixture` 会在 RGB 图中用颜色规则找到红色方块，并返回与真实 VL 模型一致的区域结构。它的作用不是替代 VL，而是先把下面这些接口固定住：

```text
1. VL 输出格式。
2. 目标区域可视化 overlay。
3. 目标区域到 depth 的索引方式。
4. depth + intrinsics + extrinsics 到世界坐标的反投影。
5. MCP 工具返回结构。
```

`openai_vision` 和 `ark_coding_vision` 是真实 VL provider。它们只负责在图上输出目标区域，不直接输出关节角或 3D 坐标。

## VL 输出格式

推荐真实 VL 模型优先输出 mask，其次输出 bbox，最低限度输出 point。

当前已支持：

```json
{
  "type": "bbox",
  "label": "red_object",
  "prompt": "pick the red block",
  "provider": "color_fixture",
  "bbox_xyxy": [199, 153, 225, 181],
  "confidence": 0.1288,
  "overlay_path": "outputs/d435i_preview/vl_region_overlay.png"
}
```

也支持 point 类型的区域输入：

```json
{
  "type": "point",
  "x": 212,
  "y": 167,
  "radius_px": 12
}
```

后续建议增加 mask：

```json
{
  "type": "mask",
  "mask_path": "outputs/vl_region/mask.png",
  "confidence": 0.91
}
```

mask 的效果通常会比 bbox 更好，因为 bbox 内可能包含桌面、夹爪、背景等无关深度。

## OpenAI Vision provider

真实 provider 使用 OpenAI Responses API。根据 OpenAI 官方文档，Responses API 支持 `input_image` 图像输入，也支持通过 `text.format` 使用 JSON schema 结构化输出。

配置文件：

```text
config/vl_providers.local.json
```

参考模板：

```text
config/vl_providers.example.json
```

`model` 可选，默认值为 `gpt-5.5`。如果后续想控制成本或延迟，可以把它换成账号可用的视觉模型。

MCP 调用示例：

```json
{
  "prompt": "pick the red block",
  "provider": "openai_vision",
  "pose": "scan",
  "width": 424,
  "height": 240
}
```

如果要使用非默认配置文件，可以额外传：

```json
{
  "config_path": "config/vl_providers.local.json"
}
```

返回的 region 会被归一化成统一格式：

```json
{
  "type": "bbox",
  "label": "red block",
  "provider": "openai_vision",
  "bbox_xyxy": [199, 153, 225, 181],
  "confidence": 0.86,
  "model": "gpt-5.5"
}
```

注意：真实 VL 模型可能框偏、框大、或选错目标。因此每次真实 VL 验证都应该查看 `overlay_path`，最终是否可用于抓取仍需要人工确认或增加更严格的自动检查。

## Ark Coding Vision provider

如果希望使用火山方舟 coding plan，可以使用 `ark_coding_vision` provider。

默认配置来自当前 coding plan 页面：

```json
{
  "providers": {
    "ark_coding_vision": {
      "api_key": "你的 API Key",
      "base_url": "https://ark.cn-beijing.volces.com/api/coding/v3",
      "model": "glm-5.2"
    }
  }
}
```

本地配置文件路径：

```text
config/vl_providers.local.json
```

MCP 调用示例：

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

该 provider 会向下面的 chat-completions endpoint 发送图像：

```text
https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions
```

注意：如果当前 coding plan 下的 `glm-5.2` 不支持图像输入，验证脚本会返回接口错误。此时说明它可以用于编程模型，但不能直接作为本项目的 VL 视觉识别模型，需要换成支持视觉输入的模型或 endpoint。

## 深度反投影逻辑

代码入口：

```text
src/perception/vl_region.py
```

核心函数：

```text
locate_red_region_fixture(...)
estimate_region_3d(...)
```

`estimate_region_3d` 会在目标区域内筛选有效深度，然后取更靠近相机的前景深度点，最后用中位数估计目标 3D 位置。当前结果更接近方块可见表面，而不是方块几何中心。对顶抓来说，这通常更有价值，因为抓取姿态应该落在可接触表面附近。

## D435i 扫描姿态

VL 识别需要相机看到足够完整的目标。原来的 `grasp` 姿态太近，容易低于 D435i 的最小有效深度；`above` 也可能只看到局部。

因此新增了 `scan` 姿态：

```powershell
.venv\Scripts\python src\sim\render_d435i_preview.py --pose scan
```

`scan` 会把夹爪根部相机移动到方块上方更适合观察的位置，主要用于视觉定位和 VL 验证。

## 自动验证

运行：

```powershell
.venv\Scripts\python src\sim\verify_vl_region.py
.venv\Scripts\python src\sim\verify_openai_vl_provider.py
.venv\Scripts\python src\sim\verify_ark_coding_vl_provider.py
```

验证内容：

```text
1. 生成 D435i scan 姿态 RGB-D 观测。
2. 用本地 color_fixture 输出 VL 风格 bbox。
3. 用 manual_region 验证调用方传入 bbox 的调试通道。
4. 生成 overlay 图，展示目标区域。
5. 在 bbox 内读取 raw depth。
6. 用相机内外参反投影到世界坐标。
7. 检查有效深度像素数量足够。
8. 检查估计位置接近红色方块所在区域。
```

典型输出文件：

```text
outputs/d435i_preview/d435i_rgb.png
outputs/d435i_preview/d435i_raw_depth.npy
outputs/d435i_preview/d435i_noisy_depth_vis.png
outputs/d435i_preview/vl_region_overlay.png
```

当前自动预检结果：

```text
target_3d center_world_m ~= [0.349, -0.549, 0.095]
status: OK
```

这里的 z 值接近方块上表面，而不是方块中心。

`verify_openai_vl_provider.py` 是可选真实模型验证：

```text
未配置 config/vl_providers.local.json 或 openai_vision.api_key: status SKIPPED
已配置 openai_vision.api_key: 调用 openai_vision，生成 OpenAI VL overlay，并检查 depth 反投影
```

`verify_ark_coding_vl_provider.py` 是可选 coding plan 验证：

```text
未配置 config/vl_providers.local.json 或 ark_coding_vision.api_key: status SKIPPED
已配置 ark_coding_vision.api_key: 调用 ark_coding_vision，生成 Ark VL overlay，并检查 depth 反投影
```

## MCP 工具

MCP server 当前新增了三个 VL/深度相关工具：

```text
vl_locate_object_region
estimate_region_3d
vl_locate_object_3d
```

典型调用意图：

```json
{
  "prompt": "pick the red block",
  "pose": "scan",
  "width": 424,
  "height": 240,
  "provider": "color_fixture"
}
```

返回内容包括：

```text
RGB-D 观测文件
VL 区域 bbox
overlay 图
3D 目标点
有效深度像素数
```

## 与真实 VL 模型的衔接方式

真实 VL provider 应该做的事情很简单：

```text
输入：RGB 图片路径 + 用户文本指令
输出：region schema，最好是 mask，其次是 bbox
```

它不应该直接输出机械臂关节角，也不应该跳过 depth。VL 模型负责“看图选择目标区域”；深度相机和几何模块负责“把区域变成 3D 位置”。

## 后续计划

建议按以下顺序推进：

```text
1. 增加 mask region 支持，减少 bbox 混入背景深度。
2. 为 OpenAI VL 结果增加多次采样/一致性检查。
3. 将 target_3d 转换成候选抓取姿态。
4. 调用现有 IK、碰撞检测和 RRT-Connect 生成抓取路径。
5. 通过 MCP 编排“看图定位 -> 规划 -> 抓取 -> GIF 验收”闭环。
```
