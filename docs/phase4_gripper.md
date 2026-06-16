# Phase 4 夹爪模型接入说明

## 当前选择

当前仓库没有可直接使用的 Dobot CR5 夹爪模型。为了先推进抓取任务闭环，本阶段先加入一个自有的简化两指平行夹爪：

```text
assets/dobot_cr5/mjcf/cr5_with_gripper.xml
```

该文件由脚本生成：

```powershell
.venv\Scripts\python src\sim\gripper_model.py
```

这样做的原因是：

```text
1. 不破坏原始 6 轴基准模型 cr5_simplified.xml。
2. 先获得可开合、可碰撞、可验证的夹爪接口。
3. 后续可以把夹爪几何替换成更高保真的外部模型。
```

## 外部候选模型

后续如果需要更真实的工业夹爪，优先候选是 Robotiq 2F-85。Google DeepMind 的 MuJoCo Menagerie 中已经有 `robotiq_2f85` MJCF 模型，包含简化机器人描述、碰撞垫、摩擦等处理，许可证为 BSD-2-Clause。

当前没有直接把它复制进仓库，是因为本阶段更重要的是先稳定：

```text
CR5 本体 + 夹爪接口 + 验证脚本 + 后续抓取技能入口
```

等 pick-and-place 任务需要更真实接触时，再做 Robotiq 2F-85 的装配、尺度/坐标系对齐、许可证登记和碰撞参数调试。

## 模型结构

新增夹爪挂在 `Link6` 的 `tool0` 位置之后：

```text
Link6
  tool0
  parallel_gripper
    gripper_mount
    gripper_tcp
    gripper_left_finger
      gripper_left_slide
    gripper_right_finger
      gripper_right_slide
```

含义：

```text
tool0         保留原有运动学语义，避免破坏已有 IK / RRT-Connect 测例。
gripper_tcp   新增夹爪末端参考点，后续抓取姿态应逐步切到这个站点。
```

夹爪有两个滑动关节：

```text
gripper_left_slide
gripper_right_slide
```

默认打开量：

```text
0.025 m
```

最大打开量：

```text
0.035 m
```

## 验证方式

运行：

```powershell
.venv\Scripts\python src\sim\verify_gripper_model.py
```

自动验证内容：

```text
1. 带夹爪模型可以被 MuJoCo 加载。
2. 模型从 6 个关节/执行器扩展为 8 个关节/执行器。
3. 左右夹爪滑动关节存在。
4. tool0、gripper_mount、gripper_tcp 站点存在。
5. 夹爪闭合和打开时，两指距离确实变大。
6. 位置执行器可以把夹爪从闭合驱动到打开。
```

可视化检查：

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py --model assets\dobot_cr5\mjcf\cr5_with_gripper.xml
```

预期现象：

```text
机械臂按原来的方式运动，夹爪在末端缓慢开合。
```

如果只想看静止打开状态：

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py --model assets\dobot_cr5\mjcf\cr5_with_gripper.xml --hold
```

## 抓取验证场景

当前已经加入第一组动力学抓取验证场景：

```text
assets/dobot_cr5/mjcf/cr5_gripper_pick_scene.xml
```

生成命令：

```powershell
.venv\Scripts\python src\sim\gripper_pick_scene.py
```

场景内容：

```text
1. 一个固定桌面，作为支撑和碰撞环境。
2. 一个红色小方块，带 free joint，可以被接触力移动和抬起。
3. 小方块设置质量、摩擦、接触求解参数。
4. 简化夹爪的摩擦垫设置较高摩擦，用来模拟橡胶夹持面。
```

自动验证命令：

```powershell
.venv\Scripts\python src\sim\verify_gripper_pick.py
```

自动验证内容：

```text
1. 生成带桌面和动态方块的抓取场景。
2. 打开夹爪并移动到方块上方。
3. 下探到抓取高度。
4. 关闭夹爪，通过接触和摩擦夹住方块。
5. 抬起机械臂，检查方块高度是否明显高于初始桌面高度。
6. 生成测试 GIF 并检查 GIF 文件有效。
```

当前自动预检结果：

```text
initial_cube_z = 0.065 m
final_cube_z   = 0.14736 m
max_cube_z     = 0.14844 m
status         = automatic pre-check passed; waiting for user GIF confirmation
```

正式 GIF：

```text
outputs/gripper_pick/simplified_gripper_pick_cube.gif
```

需要人工确认：

```text
1. 夹爪从方块上方接近，而不是横向扫过方块。
2. 两指闭合后，方块不是被瞬移，而是被夹住。
3. 抬升阶段方块跟随夹爪上升。
4. 方块没有明显穿透桌面、夹爪或地面。
5. 运动速度和停留节奏便于观察。
```

## 后续任务

下一步建议按这个顺序推进：

```text
1. 增加桌面、方块、圆柱等可抓取物体。
2. 给 gripper_tcp 定义 approach / grasp / lift / place 姿态。
3. 实现 open_gripper / close_gripper / pick / place 技能。
4. 渲染 pick-and-place GIF，由用户人工验收。
5. 如接触效果不足，再替换为 Robotiq 2F-85 高保真夹爪。
```
