# 项目后续路线图

## 1. 总体目标

本项目的长期目标不是只做一个机械臂路径规划 demo，而是逐步形成一个
可由大模型操作的虚拟机械臂系统，并为后续 VLA（Vision-Language-Action）
能力接入打基础。

目标形态：

```text
用户自然语言任务
  -> 大模型理解任务和场景
  -> 选择/组合机器人技能
  -> 感知虚拟场景状态
  -> 调用运动规划和执行器控制
  -> 在 MuJoCo 中完成可视化执行
  -> 生成可验收的 GIF / video / 日志
```

当前 RRT-Connect、碰撞检查、轨迹参数化和 GIF 验证，是后续系统的
运动执行底座。

## 2. 核心设计原则

### 2.1 大模型不直接控制关节

大模型不应直接输出低层关节角序列。更合理的分层是：

```text
大模型：
  负责理解任务、拆解步骤、选择技能、处理失败恢复。

技能层：
  负责 pick / place / push / open_gripper / scan 等可复用动作。

规划层：
  负责 IK、RRT-Connect、碰撞、奇异性、轨迹生成。

执行层：
  负责 MuJoCo 控制、夹爪动作、渲染、日志。
```

这样可以让大模型拥有“任务智能”，而不是承担不稳定的低层控制。

### 2.2 先虚拟闭环，再考虑 VLA

VLA 不应太早接入。更稳妥的顺序是：

```text
1. 先有可靠的虚拟机械臂、夹爪、相机和场景。
2. 再有结构化技能库和任务评测。
3. 再用大模型做高层任务规划。
4. 最后引入 VLA 做视觉到动作的端到端或半端到端策略。
```

VLA 需要数据、任务分布、动作接口和评测基准。当前项目应该先把这些接口
搭起来。

## 3. 阶段规划

## Phase 3 当前阶段：运动规划底座

当前已经在推进：

```text
FK / Jacobian / IK
planning collision model
RRT-Connect
shortcut
轨迹时间参数化
地面/小球/障碍物碰撞
Jacobian 奇异性裕度
GIF 验证和人工验收流程
```

下一步需要继续完善：

```text
更准确的 planning-grade collision geometry
更多障碍物形状和不可达/无解失败用例
路径导出格式
更稳定的轨迹平滑和速度曲线
```

## Phase 4：夹爪与执行器

目标：让机械臂从“移动到目标点”升级为“能操作物体”。

主要任务：

```text
1. 给 CR5 添加夹爪 MJCF 模型。
2. 定义夹爪开合关节、控制器和限位。
3. 添加可抓取物体：方块、圆柱、小工具等。
4. 实现基础 grasp pose 生成。
5. 实现 open_gripper / close_gripper / approach / retreat 技能。
6. 验证 pick-and-place 的完整闭环。
```

优先验证场景：

```text
桌面上抓取方块
抓取圆柱体
从一个区域搬运到另一个区域
夹爪接近物体时避障
抓取失败检测和恢复
```

## Phase 5：深度相机与感知

目标：让系统不是只知道脚本里的坐标，而是能从虚拟传感器感知场景。

主要任务：

```text
1. 在 MuJoCo 场景中加入 RGB-D / depth camera。
2. 渲染 RGB、depth、segmentation / object id。
3. 从 depth 生成点云或物体 3D bounding box。
4. 建立 world/camera/robot 坐标变换。
5. 输出结构化场景状态。
```

场景状态示例：

```json
{
  "objects": [
    {
      "name": "red_cube",
      "pose": [0.42, -0.30, 0.05],
      "size": [0.04, 0.04, 0.04],
      "graspable": true
    }
  ],
  "robot": {
    "gripper_open": true,
    "holding": null
  }
}
```

这一层是后续大模型和 VLA 的共同输入基础。

## Phase 6：细致场景与任务基准

目标：从单一障碍物 demo，扩展到可评测的真实任务环境。

场景方向：

```text
桌面整理
物体分类
抽屉/盒子/容器操作
障碍物遮挡
狭窄空间取物
多物体堆叠
目标物被遮挡
```

每个场景都应有：

```text
初始状态
目标状态
成功判据
失败判据
自动预检
GIF / video 人工验收
```

评价指标：

```text
任务成功率
碰撞次数
是否穿模
是否接近奇异
路径长度
执行时间
抓取成功率
重试次数
```

## Phase 7：大模型任务规划层

目标：让用户可以用自然语言操作虚拟机械臂。

大模型输入：

```text
用户指令
结构化场景状态
可用技能列表
执行反馈
失败原因
```

大模型输出不应是关节角，而应是技能调用：

```json
[
  {"skill": "scan_scene"},
  {"skill": "pick", "object": "red_cube"},
  {"skill": "place", "target": "blue_bin"}
]
```

需要实现：

```text
1. 技能 registry。
2. 大模型 tool/function calling 接口。
3. 任务执行器。
4. 执行日志和失败反馈。
5. 失败恢复策略。
```

示例技能：

```text
move_to_pose
move_to_object
open_gripper
close_gripper
pick
place
push
scan_scene
clear_obstacle
```

## Phase 8：VLA 接入准备

目标：为 Vision-Language-Action 模型接入准备数据和接口。

不要一开始就把 VLA 当成全能控制器。推荐先做两种接口：

### 8.1 VLA 作为动作建议器

```text
输入：图像 / 深度 / 语言任务 / 当前状态
输出：下一步高层动作或目标位姿
```

然后仍由现有规划器执行。

### 8.2 VLA 作为低层策略

```text
输入：RGB-D / 语言 / proprioception
输出：末端增量位姿或关节增量
```

这种方式需要更多数据和安全约束，应放在后面。

VLA 前置准备：

```text
任务数据采集
专家轨迹导出
图像/深度/动作同步记录
场景随机化
失败样本记录
统一 action space
安全过滤器
```

## 4. 推荐近期里程碑

## Milestone A：夹爪可用

完成标准：

```text
夹爪模型加载正常。
夹爪能开合。
夹爪碰撞有效。
可以抓取一个简单方块。
生成 pick-and-place GIF。
```

## Milestone B：桌面 RGB-D 场景

完成标准：

```text
场景中有桌面、多个物体、障碍物。
相机能输出 RGB 和 depth。
能从虚拟相机恢复物体位置。
机械臂能根据感知结果移动到目标物体附近。
```

## Milestone C：技能库

完成标准：

```text
实现 pick / place / move_to_object / scan_scene。
每个技能有输入、输出、失败原因。
每个技能都有自动验证和 GIF。
```

## Milestone D：大模型调用技能

完成标准：

```text
用户输入自然语言任务。
大模型生成技能序列。
执行器逐步调用技能。
失败时大模型能根据反馈重新规划。
```

## Milestone E：VLA 数据闭环

完成标准：

```text
可以记录 RGB-D、语言指令、机器人状态、动作、成功/失败标签。
可以导出统一数据集。
可以用同一接口接入 VLA 推理结果。
```

## 5. 当前下一步建议

最建议的下一步是：

```text
Phase 4：加入夹爪与基础 pick-and-place。
```

原因：

```text
1. 没有夹爪，机械臂只能“到达”，不能“操作”。
2. 大模型和 VLA 的价值主要体现在物体操作任务上。
3. 夹爪会迫使我们完善 grasp pose、接近/撤离、接触、碰撞和失败检测。
4. 夹爪任务可以自然扩展到深度相机和大模型技能调用。
```

推荐第一个可验收 demo：

```text
桌面上有一个红色方块。
机械臂从上方接近。
夹爪打开。
下降到抓取位姿。
夹爪闭合。
抬起方块。
移动到蓝色目标区域。
放下方块。
生成 GIF，用户确认。
```
## Phase 5 当前落地状态：D435i 深度相机

当前已经在简化夹爪根部加入一版 D435i 深度相机闭环：

```text
assets/dobot_cr5/mjcf/cr5_with_gripper_d435i.xml
assets/dobot_cr5/mjcf/cr5_gripper_d435i_pick_scene.xml
src/sim/d435i_model.py
src/sim/d435i_camera.py
src/sim/render_d435i_preview.py
src/sim/verify_d435i_camera.py
docs/phase5_d435i_camera.md
```

它目前完成的是感知底座，而不是完整三维重建：

```text
1. 夹爪根部 D435i 简化几何。
2. MuJoCo RGB 渲染。
3. MuJoCo depth 渲染。
4. pinhole intrinsics。
5. OpenCV 风格 world-to-camera extrinsic。
6. 深度噪声和 dropout。
7. RGB/depth 预览文件和自动预检。
```

后续应继续推进：

```text
1. 扩展 MCP 观测工具，输出点云或结构化物体列表。
2. 从 depth 生成点云。
3. 增加 segmentation / object id。
4. 建立可供大模型读取的结构化场景状态。
5. 需要高保真外观时导入 Intel RealSense ROS 的 D435/D435i mesh。
```
