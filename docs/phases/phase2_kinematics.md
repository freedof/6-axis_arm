# Phase 2: Kinematics

This phase adds a self-owned kinematics layer for the simplified Dobot CR5
MuJoCo model.

## Implemented

```text
src/robot/model.py          robot chain, joint names, limits
src/robot/kinematics.py     FK, geometric Jacobian, pose error
src/robot/ik.py             damped least-squares IK with multi-start
src/sim/verify_kinematics.py
src/sim/demo_xyz_joint_roundtrip.py
```

The implementation does not use MuJoCo for kinematics. MuJoCo is used only by
the verification script to compare `tool0` poses.

## Verify

```powershell
.venv\Scripts\python src\sim\verify_kinematics.py
```

Expected checks:

- FK position and rotation match the MuJoCo `tool0` site.
- The geometric Jacobian position rows match finite differences.
- Damped least-squares IK recovers a reachable target pose.

## Visualize xyz <-> Joint Angles

By default, the demo solves two target sphere centers and moves back and forth
between the red and blue markers:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py
```

Given a single target `x y z`, solve joint angles with position IK, then verify
the same angles with FK and animate the result in MuJoCo:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py --xyz 0.35 -0.55 0.20
```

The supplied `x y z` is the sphere center. The IK target is offset from that
center by:

```text
approach * (target_radius + tool_radius)
```

The target pose also constrains the end effector to point downward:

```text
tool0 local Z axis = world -Z axis
```

The defaults match the current MJCF visuals:

```text
target_radius = 0.014 m
tool_radius   = 0.020 m
approach      = [0, 0, 1]
```

Print only, without opening the viewer:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py --xyz 0.35 -0.55 0.20 --no-viewer
```

Given joint angles in radians, compute `x y z` and visualize that pose:

```powershell
.venv\Scripts\python src\sim\demo_xyz_joint_roundtrip.py --q 0.5 -0.8 0.7 0.4 -0.6 1.2
```

Render a GIF from a fixed front-oblique lower camera:

```powershell
.venv\Scripts\python src\sim\render_roundtrip_gif.py
```

Demo 2 keeps the red target contact vertical/downward and changes the blue
target contact to a horizontal tool orientation. At the blue target, `tool0`
local Z points along world `+X`, and the tool-center target is placed on the
sphere's negative-X side to preserve tangency:

```powershell
.venv\Scripts\python src\sim\demo_mixed_orientation_roundtrip.py
.venv\Scripts\python src\sim\render_mixed_orientation_gif.py
```

## Notes

The model uses the same joint chain as:

```text
assets/dobot_cr5/mjcf/cr5_simplified.xml
```

For this MJCF, fixed `euler="x y z"` transforms are evaluated as:

```text
Rx(x) @ Ry(y) @ Rz(z)
```

Each joint then rotates about its local z axis.
