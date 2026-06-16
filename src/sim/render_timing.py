from __future__ import annotations


DEFAULT_TARGET_DWELL_SECONDS = 1.0


def roundtrip_motion_alpha(
    frame_index: int,
    frames: int,
    fps: int,
    *,
    target_dwell_seconds: float = DEFAULT_TARGET_DWELL_SECONDS,
) -> tuple[float, bool]:
    """Return roundtrip progress with dwell at A and B.

    Returns `(alpha, reverse)`, where `alpha` is in `[0, 1]`. `reverse=False`
    means A->B, and `reverse=True` means B->A.
    """
    if frames <= 0:
        raise ValueError("frames must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    cycle_duration = frames / fps
    dwell = _effective_dwell(cycle_duration, target_dwell_seconds)
    move_duration = max(cycle_duration - 2.0 * dwell, 1e-9)
    half_move = 0.5 * move_duration
    elapsed = (frame_index % frames) / fps

    if elapsed < dwell:
        return 0.0, False
    elapsed -= dwell

    if elapsed < half_move:
        return elapsed / half_move, False
    elapsed -= half_move

    if elapsed < dwell:
        return 1.0, False
    elapsed -= dwell

    return min(elapsed / half_move, 1.0), True


def _effective_dwell(cycle_duration: float, requested_seconds: float) -> float:
    requested = max(0.0, float(requested_seconds))
    if cycle_duration <= 1e-9:
        return 0.0
    return min(requested, cycle_duration * 0.35)
