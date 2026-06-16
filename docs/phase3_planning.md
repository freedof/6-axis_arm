# Phase 3: Planning Scaffold

This phase starts the self-owned motion-planning stack on top of the existing
FK/IK and MuJoCo model.

## Implemented

```text
src/planning/collision.py      MuJoCo-backed joint-state collision checker
src/planning/rrt_connect.py    joint-space RRT-Connect and shortcutting
src/planning/singularity.py    Jacobian singularity-margin checks
src/planning/trajectory.py     joint path time parameterization
src/sim/planning_model.py      generated planning collision MJCF
src/sim/verify_planning_model.py
src/sim/planning_cases.py      representative target/orientation cases
src/sim/verify_planning.py     planning smoke test
src/sim/verify_trajectory.py   trajectory timing and validity smoke test
src/sim/demo_planned_roundtrip.py
src/sim/render_planned_roundtrip_gif.py
src/sim/obstacle_scene.py
src/sim/verify_obstacle_planning.py
src/sim/render_obstacle_planning_gif.py
src/sim/verify_render_gifs.py
```

The planner remains independent from MuJoCo. It receives a state-validity
callback and robot joint limits, while `MujocoCollisionChecker` owns the MuJoCo
model/data objects used for collision queries.

The state-validity callback used by planning verification now combines collision
validity and Jacobian singularity-margin validity. Current singularity thresholds
are:

```text
max_condition_number = 1000
min_singular_value   = 1e-3
min_manipulability   = 5e-4
```

## Collision Semantics

Planning checks use:

```text
assets/dobot_cr5/mjcf/cr5_planning.xml
```

This model is generated from `cr5_simplified.xml`. Existing display geoms stay
visible but collision-disabled, while explicit `collision_*` robot geoms are
added for planning. The target marker spheres `target_sphere` and
`target_sphere_b` are also collision-enabled, so planned paths must avoid the
balls themselves except for the intended tangent contact at the endpoint. A
`floor_collision` plane is also enabled at `z = 0 m` as the current base-plane /
ground obstacle.

The visual `floor` remains display-only, but `floor_collision` is aligned with
it so paths that visually pass through the ground are rejected by planning
validation.

Endpoint checks allow a small `1e-6 m` contact tolerance so numerical
round-off at the intended tangent contact is not reported as penetration.

`minimum_distance` can be raised above zero to treat near contacts as invalid
once obstacle scenes are added.

## Verify

```powershell
.venv\Scripts\python src\sim\verify_planning.py
```

Verify the generated planning collision model:

```powershell
.venv\Scripts\python src\sim\verify_planning_model.py
```

Expected checks:

- All representative planning cases solve IK successfully.
- Both endpoint joint states satisfy limits and are collision-free for each case.
- RRT-Connect returns a valid joint-space path for each case.
- Shortcutting preserves valid path edges for each case.
- Sampled path edges stay away from Jacobian singularities.

Verify that the renderers can produce valid animated GIFs:

```powershell
.venv\Scripts\python src\sim\verify_render_gifs.py
```

This creates small test outputs under:

```text
outputs/test_gifs/
```

The test renders 960 x 720 GIFs at 48 frames / 12 fps, then opens each GIF with
Pillow and checks format, playback duration, dimensions, non-blank first frame,
and visible frame-to-frame motion. Rendered roundtrip GIFs hold at each target
for about one second by default.

Verify the obstacle-avoidance case group:

```powershell
.venv\Scripts\python src\sim\verify_obstacle_planning.py
```

This test creates one model and one test GIF per obstacle case:

```text
outputs/obstacle_scene/tc_rrt_obstacle_*.xml
outputs/test_gifs/tc_rrt_obstacle_*_test.gif
```

Expected checks:

```text
The A/B endpoints are valid.
The direct joint-space edge is invalid because it collides with the obstacle.
RRT-Connect finds a collision-free path around the obstacle.
Shortcutting preserves valid edges.
The rendered obstacle-path GIF is valid.
```

Verify time-parameterized trajectories:

```powershell
.venv\Scripts\python src\sim\verify_trajectory.py
```

Expected checks:

```text
Raw and shortcut RRT paths are converted to smooth joint trajectories.
The sampled trajectories respect configured velocity and acceleration limits.
All sampled trajectory states remain valid under the collision checker.
All sampled trajectory states remain valid under the singularity checker.
```

## Render Planned Motion

```powershell
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py --target-dwell-seconds 1.0
```

Render a specific predefined case:

```powershell
.venv\Scripts\python src\sim\render_planned_roundtrip_gif.py --case-id TC-RRT-MULTI-005
```

Render the obstacle-avoidance demo:

```powershell
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py --all
.venv\Scripts\python src\sim\render_obstacle_planning_gif.py --all --raw-rrt
```

Default and all-case outputs:

```text
outputs/obstacle_scene/tc_rrt_obstacle_001.gif
outputs/obstacle_scene/tc_rrt_obstacle_*.gif
```

Open the live planned-path viewer:

```powershell
.venv\Scripts\python src\sim\demo_planned_roundtrip.py
.venv\Scripts\python src\sim\demo_planned_roundtrip.py --obstacle
```

Default output:

```text
outputs/roundtrip_planned_mixed_orientation.gif
```

By default, planned renderers time-parameterize the raw RRT-Connect path with
smooth cubic segment timing. Use `--shortcut` to render the shortcut-smoothed
path.

Obstacle GIF rendering uses shortcut-smoothed paths by default because raw RRT
trees are useful for debugging but can look jittery. Use `--raw-rrt` only when
inspecting the planner tree itself.

All GIF renderers use `--target-dwell-seconds 1.0` by default so the arm pauses
briefly at each target for visual inspection.

## Next Step

The next planning refinements are to improve planning-grade collision geometry,
add non-box and no-path obstacle cases, and add trajectory export/playback
utilities.
