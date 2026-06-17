# 从 RRT-Connect 到视觉抓取闭环：机械臂项目阶段进展整理

## 开始：RRT-Connect 之后，问题变了

在完成 RRT-Connect 路径规划之前，项目的核心问题是：机械臂能不能从一个姿态运动到另一个姿态，并且在这个过程中避开障碍、满足关节限制、避开奇异区域。

RRT-Connect 解决的是“会不会规划路径”的问题。它让机械臂从简单的关节插值，进入到真正的采样式路径规划：

```text
目标姿态
  -> IK 求解
  -> 碰撞检查
  -> RRT-Connect 搜索
  -> shortcut 平滑
  -> 时间参数化
  -> MuJoCo 渲染和验证
```

但这还不是抓取。RRT-Connect 阶段的目标仍然偏“几何”和“路径”：给定目标点，机械臂规划过去。进入下一阶段后，问题变成了：

```text
目标在哪里？
机械臂用什么末端执行器接触它？
相机怎样看到它？
模型怎样理解图像？
深度怎样把图像区域变成 3D 位置？
路径规划怎样接入真实抓取动作？
结果怎样交给人验收？
```

也就是说，项目从“机械臂会动”进入到“机械臂能感知并操作物体”。

## 场景分离：先把验证对象分清楚

RRT-Connect 早期使用的是红蓝目标球场景，用来验证末端接触姿态、目标切线、障碍绕行和路径规划。这些场景继续保留：

```text
assets/dobot_cr5/mjcf/cr5_simplified.xml
assets/dobot_cr5/mjcf/cr5_planning.xml
```

但抓取阶段不能再混用这些红蓝小球。抓取需要桌面、夹爪、动态物体、接触、摩擦和相机。因此后续新增了独立的抓取场景族：

```text
assets/dobot_cr5/mjcf/cr5_with_gripper.xml
assets/dobot_cr5/mjcf/cr5_with_gripper_d435i.xml
assets/dobot_cr5/mjcf/cr5_gripper_pick_scene.xml
assets/dobot_cr5/mjcf/cr5_gripper_d435i_pick_scene.xml
assets/dobot_cr5/mjcf/cr5_gripper_pick_planning.xml
```

这个分离很重要。红蓝球场景服务于路径规划验证，方块抓取场景服务于操作闭环验证。两类场景不混在一起，后续测试才不会互相污染。

## 加入夹爪：从“末端点”变成“末端执行器”

RRT-Connect 阶段，机械臂末端主要是一个 `tool0` 参考点。它适合做运动学和路径规划，但不能真正抓东西。

后续加入了简化两指平行夹爪：

```text
assets/dobot_cr5/mjcf/cr5_with_gripper.xml
```

夹爪挂在 `Link6` 的 `tool0` 之后，同时保留 `tool0`，新增 `gripper_tcp`：

```text
Link6
  tool0
  parallel_gripper
    gripper_mount
    gripper_tcp
    gripper_left_finger
    gripper_right_finger
```

保留 `tool0` 的意义是避免破坏已有 IK 和 RRT-Connect 测例；新增 `gripper_tcp` 是为了后续抓取姿态逐步转向夹爪语义。

最开始也考虑过外部高保真夹爪模型，例如 MuJoCo Menagerie 中的 Robotiq 2F-85。但本阶段优先选择简化夹爪，原因是更容易控制：

```text
1. 先跑通抓取闭环。
2. 先验证接触、摩擦、动力学和渲染。
3. 等任务需要更真实接触时，再替换高保真模型。
```

## 加入动力学抓取：抓取不再是“剧情触发”

如果没有动力学，抓取可以被写成“夹爪到了位置，方块自动跟着走”。那只是动画，不是物理交互。

当前抓取场景中，红色方块是带 `freejoint` 的动态物体，具有质量、接触、摩擦和接触求解参数。夹爪通过 position actuator 驱动，接触方块后依靠摩擦把方块夹起。

第一组抓取验证场景是：

```text
assets/dobot_cr5/mjcf/cr5_gripper_pick_scene.xml
```

它包含：

```text
固定桌面
动态红色小方块
简化两指夹爪
高摩擦夹持垫
MuJoCo 接触动力学
```

在这个阶段里，项目遇到了几个很实际的问题。

第一是夹爪几何。早期 finger 主体相对较长，而摩擦垫偏短，视觉上像是结构指在接触方块，而不是摩擦垫接触方块。后来调整为 finger 主体更短，pad 与 finger 上下平齐，让高摩擦 pad 成为主要接触面。

第二是下降高度。抓取时如果夹爪中心下降太多，手指最低点会非常接近桌面，视觉上像要顶到桌面，甚至可能出现穿刺风险。后来把抓取姿态整体抬高，当前抓取姿态下手指和 pad 底部约高于桌面 `0.055 m`。

第三是动作速度。早期抓取 GIF 中，机械臂从高位下降到抓取点时有“冲”和“过头”的感觉。后来把轨迹时间参数化按动作阶段拆开：

```text
ready_to_above: 速度较快，保持起步自然
above_to_grasp: 单独放慢，减少接近目标时的冲击
grasp_to_lift: 中等速度，便于观察抬升
```

同时，MuJoCo 仿真中不再只按 GIF 帧率更新关节目标，而是在每个 physics step 重新采样连续轨迹，减少阶跃追踪造成的抖动和过冲。

## 加入 D435i：让机械臂有自己的第一视角

有了夹爪后，下一步是让机械臂自己“看见”桌面和目标物体。因此项目在夹爪根部加入了简化版 Intel RealSense D435i：

```text
assets/dobot_cr5/mjcf/cr5_with_gripper_d435i.xml
assets/dobot_cr5/mjcf/cr5_gripper_d435i_pick_scene.xml
```

相机结构挂在 `parallel_gripper` 下：

```text
parallel_gripper
  d435i_camera_body
    d435i_depth_optical_frame
    d435i_depth
    d435i_rgb
```

当前没有直接导入外部 mesh，而是先用简化几何体，因为本阶段的重点不是相机外观，而是 RGB-D 数据链路：

```text
MuJoCo Renderer 输出 RGB
MuJoCo depth rendering 输出深度
根据 fovy 和图像尺寸计算 intrinsics
输出 world-to-camera extrinsic
加入深度噪声和 dropout
```

这一步让机械臂从“外部视角可观察”变成了“自己携带第一视角相机”。后续所有 VL 识别都基于这个 D435i RGB-D 观察。

## 视觉定位：VL 只负责找图像区域

视觉语言模型接入后，项目没有让模型直接输出关节角或 3D 坐标，而是明确分工：

```text
VL 模型：在 RGB 图上找到目标区域
D435i 深度：在目标区域内取深度
相机几何：把 2D 区域反投影到 3D 世界坐标
规划模块：把 3D 目标转成抓取姿态和路径
```

这个分工很关键。大模型擅长语义选择，但不应该绕过深度相机和几何模块。当前 VL 输出统一为 region schema，例如：

```json
{
  "type": "bbox",
  "label": "red block",
  "bbox_xyxy": [203, 137, 220, 159],
  "confidence": 0.99
}
```

随后 `estimate_region_3d` 会在 bbox 内筛选有效深度，取更靠近相机的前景点，并通过内外参反投影到世界坐标。对桌面方块来说，这个 3D 点更接近可见上表面，而不是方块几何中心。

## 从单视角到多视角：让 VL 结果更可靠

单视角识别容易出问题。目标可能太小，bbox 可能框到桌面，侧视角可能让阴影和物体混在一起。因此后续加入了高位多视角扫描：

```text
scan_high
scan_front_high
scan_left_high
scan_right_high
```

这些视角都位于方块上方或斜上方，高度约 `0.38 m`，相对方块中心偏移约 `0.055 m`。这样做的目的有两个：

```text
1. 减少低角度透视和阴影对 bbox 的干扰。
2. 让多个视角的 3D 反投影互相验证。
```

多视角流程不是把四次结果简单平均，而是先筛选：

```text
1. 每个视角分别调用 VL。
2. 每个 bbox 分别用 depth 反投影到 3D。
3. 拒绝桌面高度附近、有效深度不足或明显异常的候选。
4. 对剩余 3D 点做空间聚类。
5. 只融合落在同一目标簇的候选。
```

对外部真实 VL provider，当前规则是至少 `2` 个视角通过才进入抓取规划。只有一个视角通过时，系统会报告感知不可靠，而不是冒险抓取。

## VL 模型对比：Gemini 明显更稳

项目先尝试了火山方舟 coding plan 中的 Doubao-Seed-2.0-Pro。它可以接收图片，也能完成部分闭环，但多视角 grounding 稳定性不足，经常出现：

```text
bbox 偏大
框到桌面或背景
不同视角结果不一致
通过视角过少
```

后来通过 OpenRouter 接入 `google/gemini-3.5-flash`，效果明显提升。单视角验证中，Gemini 能准确框住小红色方块，并通过自检。多视角验证中，四个高位视角全部通过：

```text
accepted_count: 4
rejected_count: 0
fusion target: [0.34899, -0.548499, 0.095]
mean_distance_to_fused_m: 0.000189
lifted: true
status: automatic_precheck_passed
```

对应的四个 bbox 分别是：

```text
scan_high:       [203, 137, 220, 159]
scan_front_high: [203, 161, 220, 186]
scan_left_high:  [229, 138, 247, 159]
scan_right_high: [176, 138, 195, 159]
```

这说明 Gemini 不只是识别颜色，而是在不同视角下稳定地把同一个小方块作为目标，并能给出较紧的 bbox。

## target_3d 接入抓取规划：闭环真正合上

有了 `target_3d` 后，项目没有停留在“输出目标点”，而是把它接进抓取姿态生成和 RRT-Connect 规划。

对当前红色方块来说，VL + depth 得到的是上表面附近的点。因此抓取前会估计方块中心：

```text
cube_center = target_3d.center_world_m - [0, 0, cube_half_height]
```

然后生成三个关键姿态：

```text
above: 位于方块中心上方
grasp: 夹爪中心位于方块上半部，避免下降太深
lift:  抬升到方块上方
```

抓取路径分成三段规划：

```text
ready_to_above
above_to_grasp
grasp_to_lift
```

每段都经过：

```text
IK
关节限位检查
MuJoCo-backed 碰撞检查
Jacobian 奇异性裕度检查
RRT-Connect 搜索
shortcut 平滑
时间参数化
```

最终流程变成：

```text
用户指令
  -> D435i 多视角拍照
  -> VL 输出 bbox
  -> depth 反投影到 target_3d
  -> 多视角融合
  -> 估计 cube_center
  -> 生成 above/grasp/lift 姿态
  -> RRT-Connect 规划三段路径
  -> MuJoCo 动力学抓取
  -> 渲染 GIF
  -> 用户验收
```

这是 RRT-Connect 之后最重要的变化：路径规划不再是孤立能力，而是被放进了完整的感知-规划-执行闭环中。

## MCP 封装：把机械臂能力变成可调用工具

为了让 Codex 能直接操作机械臂仿真，项目把当前能力封装成了本地 MCP server：

```text
src/mcp_robot/server.py
```

它暴露的工具包括：

```text
get_robot_capabilities
get_scene_state
render_d435i_preview
vl_locate_object_region
estimate_region_3d
vl_locate_object_3d
multi_view_vl_locate_object_3d
plan_pick_from_target_3d
vl_pick_cube
multi_view_vl_pick_cube
```

这样，Codex 不只是改代码，还能调用机械臂能力：拍照、识别、反投影、规划、抓取、渲染 GIF。

后续如果希望 coding plan、Kimi-K2 或其他模型也调用这些能力，需要区分模型和 MCP host。模型本身不能直接连接本地 MCP server，必须由一个 host、client 或本地 harness 代它执行工具调用。比较稳的方式是：

```text
外部模型
  -> 本地 robot-agent harness
  -> robot MCP server
  -> MuJoCo 仿真和验证
```

这样可以保留工具白名单、日志、超时控制和用户验收边界。

## 验证方式：自动预检不等于最终通过

项目在这一阶段逐步形成了清晰的验收规则：

```text
1. 脚本和数值检查通过，只代表 automatic pre-check passed。
2. 抓取、绕障、VL 等场景必须生成对应 GIF。
3. 最终是否通过，由用户查看 GIF 后确认。
```

这个规则非常重要。因为机器人任务不是单纯单元测试，很多问题需要视觉判断：

```text
夹爪有没有穿桌面？
动作有没有明显过冲？
方块是不是被真实夹住，而不是被撞飞？
VL overlay 有没有框到正确目标？
多视角是不是选中了同一个物体？
```

因此当前文档结构也做了整理：

```text
AGENTS.md                    项目入口和规则索引
docs/validation_guide.md     验证命令和验收流程
docs/rrt_connect_test_report.md  RRT-Connect 测例和 GIF 映射
docs/phase4_gripper.md       夹爪与动力学抓取
docs/phase5_d435i_camera.md  D435i RGB-D 相机
docs/phase6_vl_perception.md VL 视觉定位
docs/vl_planned_pick.md      VL 到 RRT-Connect 抓取闭环
docs/mcp_robot_server.md     MCP 工具说明
```

## 当前阶段留下的问题

虽然闭环已经跑通，但项目仍然有几个明确问题。

第一，多物体歧义还没有解决。当前红色方块是单目标场景。下一步需要测试“同形状、同颜色、不同位置”的物体。此时不能只让 VL 输出一个 bbox，而应该让它输出候选列表，再由深度和几何模块做 3D 聚类与目标选择。

第二，bbox 仍然不是最终形态。对抓取任务来说，mask 通常比 bbox 更稳，因为 bbox 里可能混入桌面、阴影、夹爪或背景。后续应该支持 mask region，减少无关深度污染。

第三，MCP 与外部模型的关系还需要工程化。Codex 当前可以调用 MCP，但外部 coding plan 或 Kimi-K2 不能因为“模型强”就直接访问本地机械臂。需要本地 harness 或支持 MCP 的 host。

第四，当前仍是仿真抓取。虽然已经引入了接触、摩擦、质量、惯量和 actuator 控制，但如果未来接真实机械臂，还需要更严格的安全边界、执行确认、急停逻辑和真实传感器标定。

## 下一步：从单物体抓取走向场景级操作

RRT-Connect 之后，项目已经完成了一个关键转变：机械臂不再只是规划路径，而是开始拥有“看见目标、理解指令、生成抓取、执行验证”的闭环能力。

下一阶段最重要的方向是多物体场景：

```text
1. 同形状、同颜色、不同位置的多个方块。
2. VL 输出所有候选，而不是只输出一个目标。
3. 深度相机把每个候选变成 3D 位置。
4. 根据“左边/右边/最近/上一次未抓取”等任务语义选择目标。
5. 用多视角聚类验证不同视角是否选中了同一个物体。
6. 建立 scene memory，给物体分配稳定 track_id。
```

再往后，才是更完整的 VLA 方向：

```text
视觉感知
语言指令
场景记忆
动作规划
夹爪接触
任务级技能组合
```

从这个角度看，RRT-Connect 是项目的运动规划骨架；夹爪、D435i、VL、MCP 和多视角抓取，则是在这个骨架上长出的感知与操作能力。
