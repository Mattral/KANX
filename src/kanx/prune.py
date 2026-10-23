"""Shared cutoff rule for KAN edge pruning.

An edge is one input feature connected to one output feature. Callers score
each edge, then use :func:`keep_mask` to decide which edges stay.

The score is either the L2 norm of that edge's parameters, or the RMS of the
edge function on a batch of inputs. Both cutoffs below apply to whichever
score the caller computed:

* ``threshold`` — zero edges whose score is less than or equal to this value.
  The slot stays, and the contraction still visits it.
* ``relative`` — zero edges whose score is less than or equal to
  ``relative`` times the strongest edge in the same layer.

When both are set, the higher cutoff is used.
"""
from __future__ import annotations

import numpy as np


def keep_mask(
    scores,
    *,
    threshold: float | None = None,
    relative: float | None = None,
) -> np.ndarray:
    """Return a boolean mask that is True for edges that should be kept.

    ``scores`` has shape ``(in_features, out_features)``. The comparison is
    done in float64. Non-finite scores raise ``ValueError`` instead of
    zeroing an arbitrary set of edges.
    """
    if threshold is None and relative is None:
        raise ValueError("prune() needs a threshold, a relative cutoff, or both")
    if threshold is not None and float(threshold) < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if relative is not None and not 0.0 <= float(relative) <= 1.0:
        raise ValueError(f"relative must be between 0 and 1, got {relative}")

    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"edge scores must have shape (in_features, out_features), got {array.shape}")
    if array.size and not np.isfinite(array).all():
        raise ValueError("edge scores must be finite")

    cuts: list[float] = []
    if threshold is not None:
        cuts.append(float(threshold))
    if relative is not None:
        peak = float(np.max(array)) if array.size else 0.0
        cuts.append(float(relative) * peak)
    cutoff = max(cuts) if cuts else 0.0
    return array > cutoff


def prune_stats(keep: np.ndarray) -> dict[str, int]:
    """Count pruned, total, and still-active edges in a keep-mask."""
    total = int(keep.size)
    active = int(np.count_nonzero(keep))
    return {"pruned": total - active, "total": total, "active": active}
