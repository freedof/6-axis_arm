# Phase 5 D435i 深度相机接入说明

## 当前结论

可以找到合适的 D435i 几何模型来源，也可以找到可参考的代码实现。

几何模型候选：

```text
Intel RealSense ROS / realsense2_description
```

该仓库提供 RealSense 相机的 URDF/Xacro、mesh 和描述文件。其中 `realsense2_description` 目录包含 D435i 相关 Xacro，`meshes` 目录包含 D435 系列外观 mesh。许可证为 Apache-2.0。

参考链接：

```text
https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description
https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description/meshes
https://github.com/realsenseai/realsense-ros/tree/ros2-master/realsense2_description/urdf
https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/package.xml
```

当前项目没有直接拷贝外部 mesh，而是先加入一版简化 D435i 几何体：

```text
assets/dobot_cr5/mjcf/cr5_with_gripper_d435i.xml
assets/dobot_cr5/mjcf/cr5_gripper_d435i_pick_scene.xml
```

原因是本阶段更重要的是先把“夹爪根部相机 + RGB-D 渲染 + 内外参 + 深度噪声 + 验证脚本”闭环跑通。后续需要更真实外观时，再把 Intel RealSense ROS 的 mesh 按许可证登记后导入。

## 挂载位置

D435i 挂在简化夹爪根部的 `parallel_gripper` 下面：

```text
Link6
  tool0
  parallel_gripper
    gripper_mount
    gripper_tcp
    d435i_camera_body
      d435i_mount
      d435i_depth_optical_frame
      d435i_depth
      d435i_rgb
```

当前相机朝向跟随夹爪姿态。在抓取任务的预抓取位，D435i 从夹爪根部向下观察桌面、方块和夹爪指尖。

## 生成命令

生成带 D435i 的夹爪模型和抓取场景：

```powershell
.venv\Scripts\python src\sim\d435i_model.py
```

该脚本会先生成简化夹爪和抓取场景，再在夹爪根部追加 D435i。

## RGB-D 相机实现

代码入口：

```text
src/sim/d435i_camera.py
```

当前实现参考了 `E:\Projects\3d_reconstruct` 中的 D435i 仿真思路，主要包含：

```text
1. 使用 MuJoCo Renderer 从命名相机渲染 RGB。
2. 使用 MuJoCo depth rendering 输出深度图。
3. 根据相机 fovy 和图像尺寸计算 pinhole intrinsics。
4. 输出 OpenCV 风格 world-to-camera extrinsic。
5. 给深度图加入距离相关高斯噪声、边缘 dropout 和远距离 dropout。
```

当前参数：

```text
depth resolution 默认值: 848 x 480
预览验证默认值:       424 x 240
最小有效深度:         0.17 m
最大有效深度:         10.0 m
baseline:             50 mm
vertical FOV:          65 deg
```

注意：抓取瞬间相机离桌面/方块很近，可能低于 D435i 的 0.17 m 最小有效深度。因此默认验证姿态使用 `above` 预抓取观察位，而不是 `grasp` 贴近姿态。

用于视觉定位时，推荐使用 `scan` 姿态。该姿态会让夹爪根部相机移动到方块上方更适合观察的位置，便于 VL 模型看到较完整的目标区域。

## 渲染预览

生成 RGB、raw depth、noisy depth 和深度可视化图片：

```powershell
.venv\Scripts\python src\sim\render_d435i_preview.py
```

默认输出：

```text
outputs/d435i_preview/d435i_rgb.png
outputs/d435i_preview/d435i_external_context.png
outputs/d435i_preview/d435i_raw_depth.npy
outputs/d435i_preview/d435i_noisy_depth.npy
outputs/d435i_preview/d435i_raw_depth_vis.png
outputs/d435i_preview/d435i_noisy_depth_vis.png
```

其中 `d435i_rgb.png` 是夹爪根部 D435i 的真实第一视角；`d435i_external_context.png`
是外部对照视角，用来确认相机、夹爪、桌面和红色方块之间的空间关系。

可以指定不同姿态：

```powershell
.venv\Scripts\python src\sim\render_d435i_preview.py --pose ready
.venv\Scripts\python src\sim\render_d435i_preview.py --pose above
.venv\Scripts\python src\sim\render_d435i_preview.py --pose grasp
.venv\Scripts\python src\sim\render_d435i_preview.py --pose lift
.venv\Scripts\python src\sim\render_d435i_preview.py --pose scan
```

其中 `grasp` 姿态预期可能出现有效深度比例下降，这是因为相机距离桌面和方块过近，不代表渲染失败。

D435i 第一视角默认隐藏 MuJoCo site 标记，避免把黄色目标点、TCP 点等调试标记误当成真实物体。需要调试时可以在更底层渲染接口中打开 `show_sites`。

## 自动验证

运行：

```powershell
.venv\Scripts\python src\sim\verify_d435i_camera.py
.venv\Scripts\python src\sim\verify_vl_region.py
```

自动检查内容：

```text
1. 带 D435i 的 MuJoCo 场景可以加载。
2. d435i_depth 和 d435i_rgb 两个命名相机存在。
3. RGB 图片、raw depth、noisy depth、深度可视化文件都能生成。
4. intrinsics 为 3x3。
5. world-to-camera extrinsic 为 4x4。
6. raw depth 和 noisy depth 有足够有效像素。
7. VL 风格目标区域可以通过 depth 反投影成 3D 世界坐标。
```

当前自动预检结果：

```text
pose: above
raw_depth_valid_ratio:   about 0.948
noisy_depth_valid_ratio: about 0.947
status: OK
```

VL 目标区域和深度反投影的详细说明见：

```text
docs/phases/phase6_vl_perception.md
```

## 后续计划

下一步建议：

```text
1. 继续扩展 MCP 观测工具，例如输出点云或结构化物体列表。
2. 输出 RGB/depth/intrinsics/extrinsics 的结构化结果。
3. 从 depth 生成点云，并验证 camera/world/robot 坐标变换。
4. 加入 segmentation / object id，辅助大模型理解场景。
5. 在需要真实外观时导入 Intel RealSense ROS 的 D435 mesh。
```
