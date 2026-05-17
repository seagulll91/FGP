# FGP

Full-body IMU motion capture pipeline: real-time denoising, pose estimation (GlobalPose / ASIP / LIP), and Unity visualization.

## Setup

1. Clone this repository.
2. Install Python dependencies (PyTorch, `articulate`, etc.).
3. Place model weights under `checkpoints/` (not included in git due to size).
4. Place Unity builds under `Unity_exe/` (not included in git).
5. Place SMPL model under `models/` (e.g. `SMPL_male.pkl`).

## Entry scripts

| Script | Pose model |
|--------|------------|
| `online_fgp+globalpose.py` | GlobalPose Full_GR_OV |
| `online_fgp+asip.py` | ASIP |
| `online_fgp+LIP.py` | LIP (BiPoser) |

## Checkpoints (local)

Expected layout:

```
checkpoints/
  canonicalization/   # IMU denoise
  ASIP/
  LIP/
```
