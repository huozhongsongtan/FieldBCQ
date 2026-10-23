"""Runtime-only interventions on identical two-quality-head checkpoint weights."""

QUALITY_MODES = ("class", "region", "boundary", "joint")
QUALITY_FORMULAS = {
    "class": "s", "region": "s * qr", "boundary": "s * qb",
    "joint": "s * sqrt(qr * qb)",
}


def set_quality_scoring(model, mode=None, blend=None):
    """Set head attributes before validation/prediction; tolerate historical missing attributes."""
    if mode is not None and blend is not None:
        raise ValueError("Choose either quality_mode or quality_blend, not both")
    if mode is not None and mode not in QUALITY_MODES:
        raise ValueError(f"Unsupported quality_mode: {mode}")
    if blend is not None and blend not in (0, 1):
        raise ValueError("quality_blend must be 0 or 1")
    head = model.model[-1]
    if type(head).__name__ != "BASMQCSegment":
        if mode is not None or blend is not None:
            raise ValueError("Quality scoring requires a BASMQCSegment checkpoint (e.g. model/abc/best.pt)")
        return dict(quality_mode=None, quality_blend=None, quality_formula=None)
    if mode is not None:
        head.quality_mode, head.quality_blend = mode, 1.0
    elif blend is not None:
        head.quality_mode, head.quality_blend = "joint", float(blend)
    selected = getattr(head, "quality_mode", "joint")
    actual_blend = float(getattr(head, "quality_blend", 1.0))
    if selected not in QUALITY_MODES or actual_blend not in (0, 1):
        raise ValueError("Four-way evaluation requires a valid quality mode and blend of 0 or 1")
    effective = "class" if actual_blend == 0 else selected
    return dict(quality_mode=effective, quality_blend=actual_blend,
                quality_formula=QUALITY_FORMULAS[effective])
