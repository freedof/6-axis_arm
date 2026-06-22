# AGENTS.md

## Project

This workspace builds a self-owned motion-planning and perception stack for a
Dobot 6-axis arm in MuJoCo.

Core principles:

- Implement FK, IK, planning, trajectory generation, perception glue, and robot
  skills in this repository rather than delegating planning to MoveIt.
- Use MuJoCo for model verification, visualization, simulation, rendering, and
  collision/contact validation.
- Keep generated media under `outputs/`.
- Do not push to git remotes unless the user explicitly asks for a push.

## Environment

Use the workspace virtual environment:

```powershell
.venv\Scripts\python
```

Install or update dependencies with:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

Core dependencies:

```text
mujoco
numpy
Pillow
```

## Scene Families

Baseline and legacy RRT roundtrip scenes:

```text
assets/dobot_cr5/mjcf/cr5_simplified.xml
assets/dobot_cr5/mjcf/cr5_planning.xml
```

These scenes include the red/blue target spheres used by the legacy roundtrip
and RRT-Connect validation cases.

Gripper, cube-pick, D435i, and planned-pick scenes:

```text
assets/dobot_cr5/mjcf/cr5_with_gripper.xml
assets/dobot_cr5/mjcf/cr5_with_gripper_d435i.xml
assets/dobot_cr5/mjcf/cr5_gripper_pick_scene.xml
assets/dobot_cr5/mjcf/cr5_gripper_d435i_pick_scene.xml
assets/dobot_cr5/mjcf/cr5_gripper_multi_object_scene.xml
assets/dobot_cr5/mjcf/cr5_gripper_d435i_multi_object_scene.xml
assets/dobot_cr5/mjcf/cr5_gripper_pick_planning.xml
```

These scenes should not contain the legacy red/blue target spheres.

## Key Commands

Quick model and kinematics checks:

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
.venv\Scripts\python src\sim\verify_kinematics.py
```

Planning smoke checks:

```powershell
.venv\Scripts\python src\sim\verify_planning.py
.venv\Scripts\python src\sim\verify_planning_model.py
.venv\Scripts\python src\sim\verify_obstacle_planning.py
.venv\Scripts\python src\sim\verify_trajectory.py
```

Gripper, scene separation, D435i, and VL checks:

```powershell
.venv\Scripts\python src\sim\verify_gripper_model.py
.venv\Scripts\python src\sim\verify_gripper_pick.py
.venv\Scripts\python src\sim\verify_scene_separation.py
.venv\Scripts\python src\sim\verify_multi_object_scene.py
.venv\Scripts\python src\sim\verify_language_goal.py
.venv\Scripts\python src\sim\verify_d435i_camera.py
.venv\Scripts\python src\sim\verify_vl_region.py
```

Real VL provider checks are optional and depend on local credentials in
`config/vl_providers.local.json`:

```powershell
.venv\Scripts\python src\sim\verify_openai_vl_provider.py
.venv\Scripts\python src\sim\verify_ark_coding_vl_provider.py
.venv\Scripts\python src\sim\verify_openrouter_vl_provider.py
```

MCP server verification:

```powershell
.venv\Scripts\python src\mcp_robot\verify_server.py
```

Complete validation guidance lives in:

```text
docs/validation_guide.md
```

## Code Layout

```text
src/robot/model.py          CR5 chain, joint names, limits
src/robot/kinematics.py     FK, geometric Jacobian, pose error
src/robot/ik.py             Damped least-squares IK
src/robot/mujoco_compare.py MuJoCo pose comparison helper
src/planning/*.py           Collision, singularity checks, RRT-Connect, trajectories
src/perception/*.py         VL region schemas, provider calls, RGB-D back-projection
src/mcp_robot/*.py          Local MCP robot server and high-level robot skills
src/sim/*.py                Model generators, verification, demos, rendering scripts
docs/                       Phase notes, roadmap, validation reports
assets/vendor/              Downloaded vendor ROS reference model
```

## Documentation Map

Use these documents as the source of truth for details:

```text
docs/validation_guide.md        Validation command groups and acceptance rules
docs/rrt_connect_test_report.md RRT-Connect cases, GIF mappings, scene checks
docs/project_roadmap.md         Long-term roadmap toward LLM/VLA operation
docs/phase3_planning.md         Planning and obstacle validation notes
docs/phase4_gripper.md          Simplified gripper model and cube-pick notes
docs/phase5_d435i_camera.md     D435i camera model and RGB-D preview notes
docs/phase6_vl_perception.md    VL provider setup and RGB-D target localization
docs/vl_planned_pick.md         target_3d to grasp-pose and planned-pick flow
docs/mcp_robot_server.md        Local MCP tool schemas and usage examples
```

Keep case-specific validation details in the test documents, not in this file.

## Project Skill

This project includes a Codex skill for VL-assisted robot perception:

```text
.codex/skills/vl-robot-perception/SKILL.md
```

Use it when Codex should inspect D435i RGB images, act as a human-in-the-loop VL
judge, call a configured VL provider, produce a bbox/point region, or lift a VL
region into a 3D world target.

## Current State

- FK/Jacobian/IK are implemented locally and verified against MuJoCo.
- RRT-Connect planning, shortcutting, obstacle checks, singularity margins, and
  time-parameterized playback are implemented for representative cases.
- The simplified parallel gripper and cube-pick scenes are generated from code.
- The gripper-mounted D435i scene supports RGB/depth previews and noisy depth
  visualization.
- The MCP server exposes model generation, preview rendering, VL localization,
  target_3d planning, and planned cube-pick skills.
- Real VL providers include `openai_vision`, `ark_coding_vision`, and
  `openrouter_vision`; `OpenRouter + google/gemini-3.5-flash` is currently the
  strongest observed provider for the red cube multi-view grounding test.

## Implementation Rules

- Keep FK/Jacobian/IK independent from MuJoCo; use MuJoCo for verification and
  simulation.
- If the MJCF joint chain changes, update `src/robot/model.py` and rerun
  `verify_kinematics.py`.
- Keep `cr5_simplified.xml` as the 6-axis baseline. Update generators rather
  than hand-editing generated gripper or D435i MJCF files.
- Preserve the distinction between sphere-center targets and `tool0` targets in
  legacy roundtrip demos.
- For gripper pick scenes, keep table/cube/contact/friction behavior in the
  generated scene files and corresponding simulation code.
- For VL workflows, real providers should output a region schema only; depth
  and camera geometry compute the 3D target.
- Keep low-level robot logic in `src/robot`, `src/planning`, `src/perception`,
  and `src/sim`; keep high-level orchestration in `src/mcp_robot`.

## Validation Policy

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

Before asking for user acceptance, generate or refresh the relevant GIFs under
`outputs/` and report:

```text
case id
case name
scene description
what should be visually confirmed
GIF path
automatic pre-check status
user acceptance status: pending
```

## Next Work

Use `docs/project_roadmap.md` as the roadmap source. The near-term VL direction
is multi-object validation: same shape, same color, different positions,
candidate-list grounding, 3D clustering, target selection, and scene memory.
