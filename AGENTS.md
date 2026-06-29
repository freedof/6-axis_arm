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
.venv\Scripts\python src\sim\verify_multi_object_vl_pick_place.py
.venv\Scripts\python src\sim\verify_collect_cylinders_to_tray.py
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
docs/validation/validation_guide.md
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
docs/                       Roadmap, phase notes, validation reports, articles
assets/vendor/              Downloaded vendor ROS reference model
```

## Documentation Map

Use these documents as the source of truth for details:

```text
docs/validation/validation_guide.md        Validation command groups and acceptance rules
docs/validation/rrt_connect_test_report.md RRT-Connect cases, GIF mappings, scene checks
docs/project_roadmap.md                    Long-term roadmap toward LLM/VLA operation
docs/phases/phase3_planning.md             Planning and obstacle validation notes
docs/phases/phase4_gripper.md              Simplified gripper model and cube-pick notes
docs/phases/phase5_d435i_camera.md         D435i camera model and RGB-D preview notes
docs/phases/phase6_vl_perception.md        VL provider setup and RGB-D target localization
docs/vl_planned_pick.md                    target_3d to grasp-pose and planned-pick flow
docs/mcp_robot_server.md                   Local MCP tool schemas and usage examples
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

## Codex-Orchestrated VL Dispatch

For user-facing manipulation requests, Codex should parse the task intent first,
then call the virtual arm MCP tools with explicit structured arguments. Do not
use the local `parse_language_goal` rule parser as the primary interpreter for
Chinese or open-ended instructions; keep it as a compatibility fallback and
smoke-test helper.

When a request requires selecting or localizing an object from the D435i scene,
Codex should mark the task as requiring VL grounding, choose a provider
explicitly, and pass a Codex-parsed `language_goal` into `multi_object_vl_locate`
or `language_multi_view_pick_and_place`. Use real VL providers such as
`openrouter_vision`, `openai_vision`, or `ark_coding_vision` for model-based
recognition; use `codex_vision` only for interactive Codex-in-the-loop bbox
selection; use `color_fixture` only for deterministic local validation.

Keep VL pre-grounding free of god's-eye knowledge. Before a real or
Codex-in-the-loop VL provider has produced visual regions and depth-lifted 3D
targets, Codex must not inspect or use scene object specs, MJCF body names,
default object lists, generated scene metadata, or MuJoCo state to determine
which objects exist, how many targets there are, their colors, names, positions,
or which objects match the user's instruction. For open-ended manipulation
requests such as "pick all cubes into the tray", "pick the red cube into the
tray", "pick all cubes except the red one", or "pick the red cube and green
cylinder", Codex should pass an open visual `target_query` plus the destination
into the live/VL pipeline and let D435i RGB-D + VL grounding discover the target
instances. After VL grounding succeeds, the resulting observed targets
(`vl_target_01`, `vl_target_02`, etc.) are the legitimate target set for
counting, ordering, tray-slot assignment, parallel planning, and execution.

For live demonstrations, use `src/sim/launch_live_robot_session.py` first. It
reuses an existing waiting session when provider/model match, otherwise starts
`src/sim/live_robot_session.py` as the long-running MuJoCo viewer process and
waits until `outputs/live_session/status.json` reports `status: waiting`.

Preferred OpenRouter startup:

```powershell
.venv\Scripts\python src\sim\launch_live_robot_session.py --provider openrouter_vision --model google/gemini-3.5-flash
```

The expected flow is:

```text
1. User: "启动机械臂并加载场景"
   Codex starts the live session; MuJoCo shows the arm and multi-object scene,
   static and waiting for a command.
2. User: manipulation instruction
   Codex parses the instruction into a structured JSON command and submits it
   atomically with `src/sim/submit_live_robot_command.py`.
3. The live session moves to scan poses, captures D435i views, runs VL/depth
   localization, plans and executes the pick/place motion in the same viewer.
4. After the task completes, the session waits 5 seconds and closes.
```

Live command safety rules:

```text
Use `src/sim/submit_live_robot_command.py` or an equivalent temp-file +
atomic replace operation to create `outputs/live_session/command.json`.
Never use PowerShell `Set-Content` directly on `command.json`.
After submitting, do not read/open `command.json`; the live session deletes it
after parsing, and Windows file locks can otherwise close the MuJoCo session.
Monitor progress via `outputs/live_session/status.json`, `live_stdout.log`,
and `live_stderr.log` only.
```

Default live command fast path:

```text
For routine manipulation instructions in an already running live session, keep
the Codex control loop minimal unless the user asks for deeper inspection:

1. Reuse the current live session and check `outputs/live_session/status.json`
   once to confirm `status: waiting`.
2. Build the minimal structured command directly from the user's instruction.
3. Submit via `src/sim/submit_live_robot_command.py --command-b64` so Chinese
   instructions are transported as explicit UTF-8.
4. Monitor only `outputs/live_session/status.json` until completion.
5. Open overlays, GIFs, stdout, stderr, or planning summaries only when the
   command fails, the VL result is ambiguous, or the user asks to inspect them.

Do not re-read broad docs or source files before every live command when the
existing command schema and session state are already known.
```

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
