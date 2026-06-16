# Phase 1: Dobot CR5 MuJoCo Model Setup

This phase establishes a loadable MuJoCo model for a Dobot 6-axis arm and a
small verification path for motion smoke testing.

## Source Model

The vendor ROS model is downloaded into:

```text
assets/vendor/TCP-IP-ROS-6AXis
```

The CR5 source files used as references are:

```text
assets/vendor/TCP-IP-ROS-6AXis/dobot_description/urdf/cr5_robot.urdf
assets/vendor/TCP-IP-ROS-6AXis/dobot_description/meshes/cr5
```

The official CR5 visual meshes are DAE files. The first MuJoCo milestone uses a
simplified MJCF model instead of mesh visuals so that the simulator, joint
chain, actuators, and scripts are stable before we invest in mesh conversion and
collision cleanup.

## Generated MuJoCo Model

```text
assets/dobot_cr5/mjcf/cr5_simplified.xml
```

The MJCF keeps the CR5 joint names, joint origins, axes, and position limits
from the vendor URDF:

```text
joint1 [-3.14,  3.14]
joint2 [-3.14,  3.14]
joint3 [-2.86,  2.86]
joint4 [-3.14,  3.14]
joint5 [-3.14,  3.14]
joint6 [-6.28,  6.28]
```

The visual geometry is intentionally simplified with cylinders and capsules. The
geoms have collision disabled in this phase; planning-grade collision bodies are
the next pass.

## Verify

Install dependencies inside the workspace virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Run the headless smoke test:

```powershell
.venv\Scripts\python src\sim\verify_cr5_model.py
```

Open the interactive viewer:

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py
```

Use `--hold` to keep the ready pose steady:

```powershell
.venv\Scripts\python src\sim\view_cr5_model.py --hold
```

## Phase 1 Acceptance

- MuJoCo loads the MJCF without XML/compiler errors.
- The model exposes 6 joints and 6 position actuators.
- `tool0` site moves when joint targets change.
- The viewer can show the simplified CR5 arm in home/ready/sweeping poses.

## Next Step

The next useful refinement is to add planning-grade collision geometry. After
that, the same model can serve the FK/IK and RRT-Connect work without changing
the simulator interface.
