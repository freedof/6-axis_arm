# Dobot 6-Axis Arm Motion Planning in MuJoCo

This workspace is building a self-owned motion-planning stack for a Dobot
6-axis arm in MuJoCo. MuJoCo is used for simulation, visualization, and collision
queries; kinematics, IK, path planning, and trajectory generation will be
implemented in this repository.

Current milestone:

```text
Phase 3: MuJoCo-backed collision checking and joint-space planning scaffold
```

Start with:

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
```

Verify kinematics with:

```powershell
.venv\Scripts\python src\sim\verify_kinematics.py
```

Verify the first planning layer with:

```powershell
.venv\Scripts\python src\sim\verify_planning.py
```

Verify the generated planning collision model:

```powershell
.venv\Scripts\python src\sim\verify_planning_model.py
```

The planning verification now runs multiple representative target and
orientation cases defined in:

```text
src/sim/planning_cases.py
```

Planning checks use `assets/dobot_cr5/mjcf/cr5_planning.xml`, generated from
the display model with visual geoms collision-disabled and explicit
`collision_*` geoms plus `target_sphere`, `target_sphere_b`, and
`floor_collision` enabled.

Planning verification also checks Jacobian singularity margins. Current paths
must satisfy the thresholds in `src/planning/singularity.py` for condition
number, minimum singular value, and manipulability.

Verify that demo renderers can generate valid GIF files with:

```powershell
.venv\Scripts\python src\sim\verify_render_gifs.py
```

The test GIFs use 960 x 720 resolution, 48 frames, and 12 fps, so each
verification animation lasts about 4 seconds for easier visual inspection.
Rendered roundtrip GIFs pause at each target for about 1 second by default.

Verify obstacle avoidance, including a rendered GIF smoke test:

```powershell
.venv\Scripts\python src\sim\verify_obstacle_planning.py
```

Generate and verify the CR5 model with the simplified parallel gripper:

```powershell
.venv\Scripts\python src\sim\gripper_model.py
.venv\Scripts\python src\sim\verify_gripper_model.py
```

Generate a dynamic pick scene and verify the simplified gripper can lift a
small cube through contact and friction:

```powershell
.venv\Scripts\python src\sim\gripper_pick_scene.py
.venv\Scripts\python src\sim\verify_gripper_pick.py
```

Render the simplified-gripper pick GIF:

```powershell
.venv\Scripts\python src\sim\render_gripper_pick_gif.py
```

Run and verify the local CR5 MCP server:

```powershell
.venv\Scripts\python src\mcp_robot\server.py
.venv\Scripts\python src\mcp_robot\verify_server.py
```

Render all obstacle-avoidance GIFs for visual acceptance:

```powershell
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py --all
```

Verify path shortcutting and time-parameterized joint trajectories:

```powershell
.venv\Scripts\python src\sim\verify_trajectory.py
```

Open the viewer with:

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py
```

Open the viewer with the gripper-equipped model:

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py --model assets\dobot_cr5\mjcf\cr5_with_gripper.xml
```

Visualize two-target roundtrip motion with:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py
```

The target coordinates are sphere centers. The script offsets the IK target by
the sphere radius plus the tool-tip capsule radius so the surfaces are tangent,
and constrains the tool local Z axis to point downward at each target.

Visualize a single `xyz -> joint angles -> xyz` target with:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py --xyz 0.35 -0.55 0.20
```

Render the two-target roundtrip to GIF with:

```powershell
.venv\Scripts\python src\sim\render_roundtrip_gif.py
```

Run demo 2, where the red target is contacted downward and the blue target is
contacted horizontally with `tool0` local Z pointing along world `+X`:

```powershell
.venv\Scripts\python src\sim\demo_mixed_orientation_roundtrip.py
.venv\Scripts\python src\sim\render_mixed_orientation_gif.py
```

Render the mixed-orientation roundtrip using the current RRT-Connect path:

```powershell
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py --target-dwell-seconds 1.0
```

Render a predefined planning test case by id:

```powershell
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py --case-id TC-RRT-MULTI-005
```

Render the obstacle-avoidance planning demo:

```powershell
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py
```

Open a live viewer driven by a time-parameterized planned trajectory:

```powershell
.venv\Scripts\python src\sim\demo_planned_roundtrip.py
.venv\Scripts\python src\sim\demo_planned_roundtrip.py --obstacle
```

Demo 2 GIF:

```text
outputs/roundtrip_mixed_orientation.gif
outputs/roundtrip_planned_mixed_orientation.gif
outputs/obstacle_scene/tc_rrt_obstacle_*.gif
outputs/gripper_pick/simplified_gripper_pick_cube.gif
```

See:

```text
docs/phases/phase1_model_setup.md
docs/phases/phase2_kinematics.md
docs/phases/phase3_planning.md
docs/phases/phase4_gripper.md
docs/mcp_robot_server.md
```
