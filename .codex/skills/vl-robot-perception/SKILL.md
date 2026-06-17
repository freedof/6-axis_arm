---
name: vl-robot-perception
description: Use for this Dobot CR5 MuJoCo project when Codex needs to locate a manipulation target from the gripper-mounted D435i RGB-D camera, use Codex itself as a human-in-the-loop VL judge, call a configured VL provider such as color_fixture/manual_region/openai_vision/ark_coding_vision, convert a 2D bbox or point into a 3D world target with depth, or report VL overlay/validation results before grasp planning.
---

# VL Robot Perception

Use this skill for the `F:\6-axis arm` project when the user asks Codex to find an object visually, use a VL model, inspect D435i images, produce a bbox/point, or prepare a 3D target for grasping.

## Workflow

1. Render or reuse a D435i observation.
   - Default command:
     ```powershell
     .venv\Scripts\python src\sim\render_d435i_preview.py --pose scan
     ```
   - Default RGB path:
     ```text
     outputs/d435i_preview/d435i_rgb.png
     ```
   - Use `scan` for VL localization because it gives a clearer view and avoids the close-range depth issue of `grasp`.

2. Choose the provider.
   - `color_fixture`: deterministic local validation for the red cube.
   - `manual_region`: Codex inspects the image in the conversation and supplies bbox/point coordinates.
   - `openai_vision`: calls the OpenAI Responses API using `config/vl_providers.local.json`.
   - `ark_coding_vision`: calls the Ark coding OpenAI-compatible endpoint using `config/vl_providers.local.json`.
   - To configure real providers, copy `config/vl_providers.example.json` to `config/vl_providers.local.json` and fill in the API key. Do not commit the local file.

3. For Codex-in-the-loop VL, inspect the RGB image and return a manual region.
   - Prefer bbox for rectangular objects:
     ```json
     {
       "type": "bbox",
       "label": "red block",
       "bbox_xyxy": [199, 153, 225, 181],
       "confidence": 1.0
     }
     ```
   - Use point when only a target click is obvious:
     ```json
     {
       "type": "point",
       "label": "target object",
       "x": 212,
       "y": 167,
       "radius_px": 12,
       "confidence": 1.0
     }
     ```

4. Convert the region to a 3D target.
   - Use MCP tool `vl_locate_object_3d` when available.
   - For Codex-in-the-loop:
     ```json
     {
       "prompt": "pick the red block",
       "provider": "manual_region",
       "manual_region": {
         "type": "bbox",
         "label": "red block",
         "bbox_xyxy": [199, 153, 225, 181],
         "confidence": 1.0
       },
       "pose": "scan",
       "width": 424,
       "height": 240
     }
     ```
   - For a real VL model:
     ```json
     {
       "prompt": "pick the red block",
       "provider": "openai_vision",
       "pose": "scan",
       "width": 424,
       "height": 240
     }
     ```
   - To continue from a known 3D target into grasp planning and simulation, use
     MCP tool `plan_pick_from_target_3d`:
     ```json
     {
       "target_3d": {
         "center_world_m": [0.349, -0.549, 0.095]
       },
       "render_gif": true
     }
     ```
   - To run the whole closed loop from image capture through planned pick, use
     MCP tool `vl_pick_cube`:
     ```json
     {
       "prompt": "pick the red block",
       "provider": "color_fixture",
       "pose": "scan",
       "render_gif": true
     }
     ```
   - For a more robust closed loop, especially when one view may include table
     pixels or when multiple similar cubes are present, use
     `multi_view_vl_pick_cube`:
     ```json
     {
       "prompt": "pick the red block",
       "provider": "color_fixture",
       "poses": ["scan", "scan_left", "scan_right", "scan_high"],
       "max_parallel_vl": 4,
       "render_gif": true
     }
     ```
     The scan poses are deliberately elevated: `scan`, `scan_left`, and
     `scan_right` are about 0.30 m above the cube center, while `scan_high` is
     about 0.38 m above it. Prefer these elevated poses when low-angle
     perspective makes the VL bbox too large.
   - For the user's Ark coding plan:
     ```json
     {
       "prompt": "pick the red block",
       "provider": "ark_coding_vision",
       "model": "glm-5.2",
       "pose": "scan",
       "width": 424,
       "height": 240
     }
     ```

5. Always report the result as an automatic pre-check, not final grasp acceptance.
   - Include the overlay path.
   - Include bbox/point, 3D world target, depth value, and valid depth pixel count.
   - State whether the provider was `color_fixture`, `manual_region`, `openai_vision`, or `ark_coding_vision`.
   - If a real provider was skipped because no local config/API key exists, say so plainly.

## Validation

Run local validation before claiming the pipeline works:

```powershell
.venv\Scripts\python src\sim\verify_vl_region.py
.venv\Scripts\python src\mcp_robot\verify_server.py
```

Optional real-provider validation:

```powershell
.venv\Scripts\python src\sim\verify_openai_vl_provider.py
.venv\Scripts\python src\sim\verify_ark_coding_vl_provider.py
```

Expected no-config behavior:

```text
status: SKIPPED
reason: config/vl_providers.local.json is missing or openai_vision.api_key is not set.
```

Expected Ark no-config behavior:

```text
status: SKIPPED
reason: config/vl_providers.local.json is missing or ark_coding_vision.api_key is not set.
```

## Interpretation Rules

- Treat Codex as a usable VL judge for interactive debugging, visual acceptance, and manual bbox/point selection.
- Do not present Codex-in-the-loop VL as an autonomous runtime vision service.
- Treat `target_3d.center_world_m` as a visible-surface target estimate, not a final grasp pose.
- Convert the 3D target into a grasp pose before planning.
- For the current cube scene, convert the visible top-surface target into a cube-center estimate before top grasping.
- Prefer `vl_pick_cube` when the user wants the full instruction -> D435i -> VL -> depth -> RRT-Connect -> pick flow.
- Prefer `multi_view_vl_pick_cube` when robustness matters. It renders multiple
  views first, calls the VL provider in parallel, rejects table-height or
  inconsistent 3D candidates, and fuses only candidates in the same 3D cluster.
- In multi-object scenes with same-shaped cubes, make the prompt identify the
  intended object by color, spatial relation, or task role. Do not average
  candidates from different 3D clusters; report ambiguity instead.
- For generated visual evidence, show the overlay image in the final response when possible.
