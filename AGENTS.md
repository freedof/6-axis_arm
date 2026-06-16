# AGENTS.md

## Project

This workspace builds a self-owned motion-planning stack for a Dobot 6-axis arm
in MuJoCo.

Current model:

```text
assets/dobot_cr5/mjcf/cr5_simplified.xml
```

Generated gripper model:

```text
assets/dobot_cr5/mjcf/cr5_with_gripper.xml
```

Generated gripper + D435i models:

```text
assets/dobot_cr5/mjcf/cr5_with_gripper_d435i.xml
assets/dobot_cr5/mjcf/cr5_gripper_d435i_pick_scene.xml
```

The current robot is a simplified Dobot CR5 model derived from the vendor ROS
URDF joint chain. MuJoCo is used for visualization, simulation, rendering, and
future collision queries. Kinematics, IK, path planning, and trajectory
generation should be implemented in this repository rather than delegated to
MoveIt or other planning frameworks.

## Environment

Use the workspace virtual environment:

```powershell
.venv\Scripts\python
```

Install/update dependencies with:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

Core dependencies:

```text
mujoco
numpy
Pillow
```

## Important Commands

Verify the MuJoCo model:

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
```

Verify FK, Jacobian, and IK:

```powershell
.venv\Scripts\python src\sim\verify_kinematics.py
```

Open the live MuJoCo viewer:

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py
```

Generate and verify the gripper-equipped model:

```powershell
.venv\Scripts\python src\sim\gripper_model.py
.venv\Scripts\python src\sim\verify_gripper_model.py
```

Generate and verify the simplified-gripper cube-pick scene:

```powershell
.venv\Scripts\python src\sim\gripper_pick_scene.py
.venv\Scripts\python src\sim\verify_gripper_pick.py
```

Render the simplified-gripper cube-pick GIF:

```powershell
.venv\Scripts\python src\sim\render_gripper_pick_gif.py
```

Generate and verify the gripper-mounted D435i camera scene:

```powershell
.venv\Scripts\python src\sim\d435i_model.py
.venv\Scripts\python src\sim\verify_d435i_camera.py
```

Render RGB/depth previews from the gripper-mounted D435i:

```powershell
.venv\Scripts\python src\sim\render_d435i_preview.py
.venv\Scripts\python src\sim\render_d435i_preview.py --pose lift
.venv\Scripts\python src\sim\render_d435i_preview.py --pose scan
```

Verify VL-style region localization and D435i depth back-projection:

```powershell
.venv\Scripts\python src\sim\verify_vl_region.py
```

Run and verify the local CR5 MCP server:

```powershell
.venv\Scripts\python src\mcp_robot\server.py
.venv\Scripts\python src\mcp_robot\verify_server.py
```

Open the gripper-equipped model:

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py --model assets\dobot_cr5\mjcf\cr5_with_gripper.xml
```

Run the two-target roundtrip demo:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py
```

Render the roundtrip demo to GIF:

```powershell
.venv\Scripts\python src\sim\render_roundtrip_gif.py
```

Run the mixed-orientation demo, where red is contacted downward and blue is
contacted horizontally with `tool0` local Z pointing along world `+X`:

```powershell
.venv\Scripts\python src\sim\demo_mixed_orientation_roundtrip.py
.venv\Scripts\python src\sim\render_mixed_orientation_gif.py
```

Run the current planning smoke test:

```powershell
.venv\Scripts\python src\sim\verify_planning.py
```

Verify the generated planning collision model:

```powershell
.venv\Scripts\python src\sim\verify_planning_model.py
```

Planning verification uses representative target/orientation cases from:

```text
src/sim/planning_cases.py
```

Render the mixed-orientation roundtrip with the current RRT-Connect path:

```powershell
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py
```

Render a predefined planning case:

```powershell
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py --case-id TC-RRT-MULTI-005
```

Run GIF render verification tests:

```powershell
.venv\Scripts\python src\sim\verify_render_gifs.py
```

Run obstacle-avoidance planning verification:

```powershell
.venv\Scripts\python src\sim\verify_obstacle_planning.py
```

Run trajectory time-parameterization verification:

```powershell
.venv\Scripts\python src\sim\verify_trajectory.py
```

Render the obstacle-avoidance planning demo:

```powershell
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py --all
```

Open the live planned-path viewer:

```powershell
.venv\Scripts\python src\sim\demo_planned_roundtrip.py
.venv\Scripts\python src\sim\demo_planned_roundtrip.py --obstacle
```

Generated media:

```text
outputs/roundtrip_touch_tool_down.gif
outputs/roundtrip_mixed_orientation.gif
outputs/roundtrip_planned_mixed_orientation.gif
outputs/obstacle_scene/tc_rrt_obstacle_*.gif
outputs/gripper_pick/simplified_gripper_pick_cube.gif
outputs/d435i_preview/d435i_rgb.png
outputs/d435i_preview/d435i_noisy_depth_vis.png
```

## Code Layout

```text
src/robot/model.py          CR5 chain, joint names, limits
src/robot/kinematics.py     FK, geometric Jacobian, pose error
src/robot/ik.py             Damped least-squares IK
src/robot/mujoco_compare.py MuJoCo pose comparison helper
src/planning/*.py           Collision, singularity checks, and joint-space planning
src/sim/*.py                Verification, model generators, viewer, demo, rendering scripts
docs/                       Phase notes
assets/vendor/              Downloaded vendor ROS reference model
```

## Test Documentation

Detailed validation cases, GIF mappings, scene descriptions, automatic
pre-checks, and user-acceptance criteria are documented in:

```text
docs/rrt_connect_test_report.md
```

Keep case-specific details in that test document, not in this AGENTS.md file.

## Roadmap

The long-term project roadmap is documented in:

```text
docs/project_roadmap.md
docs/phase4_gripper.md
docs/phase5_d435i_camera.md
docs/phase6_vl_perception.md
docs/mcp_robot_server.md
```

It covers the intended direction toward grippers, depth cameras, detailed
scenes, large-model skill orchestration, and later VLA integration.

## Current Motion Semantics

For the roundtrip demo, target coordinates are sphere centers, not tool-center
positions. The script offsets the IK target so the tool tip and target sphere
are tangent:

```text
tool0_target = sphere_center + approach * (target_radius + tool_radius)
```

Defaults:

```text
target_radius = 0.014 m
tool_radius   = 0.020 m
approach      = [0, 0, 1]
```

Demo 1 constrains both target contacts so that the `tool0` local Z axis points
downward:

```text
tool0 local Z axis = world -Z axis
```

Demo 2 keeps the red target contact downward, but constrains the blue target to
a horizontal contact:

```text
red target:  tool0 local Z axis = world -Z axis
blue target: tool0 local Z axis = world +X axis
```

For the blue target in Demo 2, the tool-center target is placed on the sphere's
negative-X side so the tool can point toward `+X` while remaining tangent.

The legacy demo path between the two target poses is joint-space interpolation.
The planned demos use RRT-Connect, collision checking, shortcutting, and
time-parameterized playback.

In planning validation, the visible target spheres are collision-enabled
environment obstacles. The tool target is still tangent to each sphere, with a
small numerical contact tolerance for the intended endpoint contact.

## Implementation Notes

- Keep the FK/Jacobian/IK implementation independent from MuJoCo. Use MuJoCo
  only to verify poses and to render/simulate.
- The MJCF fixed `euler="x y z"` transforms in the current chain are matched by
  `Rx(x) @ Ry(y) @ Rz(z)` in `src/robot/kinematics.py`.
- Each joint rotates about its local z axis after the fixed transform.
- When changing the MJCF joint chain, update `src/robot/model.py` and rerun
  `verify_kinematics.py`.
- Preserve the current distinction between sphere-center targets and `tool0`
  targets.
- Keep `cr5_simplified.xml` as the 6-axis baseline model. Generate
  `cr5_with_gripper.xml` with `src/sim/gripper_model.py`; do not hand-edit the
  generated gripper model unless the generator is updated too.
- The gripper-equipped model preserves `tool0` and adds `gripper_tcp` for
  future grasping tasks.
- The first dynamic gripper validation scene is generated by
  `src/sim/gripper_pick_scene.py`; it contains a table and a free-joint cube
  with friction/contact settings for simplified-gripper lifting tests.
- The gripper-mounted D435i model is generated by `src/sim/d435i_model.py`.
  Current D435i geometry is simplified, while `docs/phase5_d435i_camera.md`
  records the external RealSense ROS mesh candidate and RGB-D validation flow.
- VL-style object localization is documented in
  `docs/phase6_vl_perception.md`. The current local `color_fixture` provider
  is only an integration-test stand-in; real VL providers should return the
  same region schema and let depth/camera geometry compute 3D target points.
- The first local MCP server lives in `src/mcp_robot/server.py`. It exposes
  high-level tools such as `get_robot_capabilities`, `get_scene_state`, and
  `pick_cube`; it also exposes D435i/VL tools such as
  `render_d435i_preview`, `vl_locate_object_region`, `estimate_region_3d`, and
  `vl_locate_object_3d`. Keep low-level robot logic in `src/robot`,
  `src/planning`, `src/perception`, and `src/sim`.
- Keep planning validation aware of Jacobian singularity margins. Current
  thresholds live in `src/planning/singularity.py`.
- For generated media, write under `outputs/`.

## Validation Expectations

Validation has two layers:

```text
1. Automatic pre-check: scripts and numerical/GIF-file checks pass.
2. User acceptance: the user visually reviews the generated GIF for each
   validation case and explicitly confirms it.
```

Do not report a validation case as finally passed only because scripts returned
`status: OK`. Report it as:

```text
automatic pre-check passed; waiting for user GIF confirmation
```

The final acceptance decision belongs to the user.

Before reporting kinematics or model changes as complete, run:

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
.venv\Scripts\python src\sim\verify_kinematics.py
```

Before reporting demo/rendering changes as complete, also run:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py --no-viewer
.venv\Scripts\python src\sim\render_roundtrip_gif.py
.venv\Scripts\python src\sim\demo_mixed_orientation_roundtrip.py --no-viewer
.venv\Scripts\python src\sim\render_mixed_orientation_gif.py
.venv\Scripts\python src\sim\verify_planning.py
.venv\Scripts\python src\sim\verify_planning_model.py
.venv\Scripts\python src\sim\verify_render_gifs.py
.venv\Scripts\python src\sim\verify_obstacle_planning.py
.venv\Scripts\python src\sim\verify_trajectory.py
.venv\Scripts\python src\sim\verify_vl_region.py
```

Before asking for user acceptance, generate or refresh GIFs for every validation
case under `outputs/`. Use the test documentation as the source of truth for
case IDs, scene descriptions, expected visual checks, and GIF paths:

```text
docs/rrt_connect_test_report.md
```

Rendered validation GIFs should pause at each target for about 1 second by
default. Use `--target-dwell-seconds` when a different pause length is needed.

Then report each GIF with:

```text
case id
case name
scene description
what should be visually confirmed
absolute or workspace-relative GIF path
automatic pre-check status
user acceptance status: pending
```

Expected current behavior:

- FK matches MuJoCo `tool0` pose to numerical precision.
- Jacobian position rows match finite differences.
- IK reaches target poses with micrometer-scale position error.
- Demo 1 targets are tangent to the tool tip and the tool points downward at
  each target.
- Demo 2 keeps the red target downward and the blue target horizontal with
  `tool0` local Z pointing along world `+X`.

## Next Likely Work

The next major phase is planning support:

```text
1. Add planning-grade collision geometry.
2. Implement a MuJoCo-backed collision checker.
3. Implement joint-space RRT-Connect.
4. Smooth and time-parameterize planned paths.
5. Replace direct interpolation in the demo with planned collision-free paths.
```
