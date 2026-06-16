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

当前仓库中没有直接调用真实在线 VL 模型。为了先验证工程接口，项目加入了一个本地可重复的 `color_fixture` provider：

```text
src/perception/vl_region.py
```

这个 provider 会在 RGB 图中用颜色规则找到红色方块，并返回与真实 VL 模型一致的区域结构。它的作用不是替代 VL，而是先把下面这些接口固定住：

```text
1. VL 输出格式。
2. 目标区域可视化 overlay。
3. 目标区域到 depth 的索引方式。
4. depth + intrinsics + extrinsics 到世界坐标的反投影。
5. MCP 工具返回结构。
```

后续接真实 VL 模型时，只需要新增 provider，让它返回同样的 region schema。

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
```

验证内容：

```text
1. 生成 D435i scan 姿态 RGB-D 观测。
2. 用本地 color_fixture 输出 VL 风格 bbox。
3. 生成 overlay 图，展示目标区域。
4. 在 bbox 内读取 raw depth。
5. 用相机内外参反投影到世界坐标。
6. 检查有效深度像素数量足够。
7. 检查估计位置接近红色方块所在区域。
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
1. 增加真实 VL provider 接口，例如 OpenAI Vision / 本地 VLM / GroundingDINO + SAM。
2. 增加 mask region 支持，减少 bbox 混入背景深度。
3. 将 target_3d 转换成候选抓取姿态。
4. 调用现有 IK、碰撞检测和 RRT-Connect 生成抓取路径。
5. 通过 MCP 编排“看图定位 -> 规划 -> 抓取 -> GIF 验收”闭环。
```
