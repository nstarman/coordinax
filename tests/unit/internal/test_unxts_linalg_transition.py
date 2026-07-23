"""coordinax's quantity-matrix machinery is provided by unxts.linalg (unxt v2)."""

__all__: tuple[str, ...] = ()

import jax.numpy as jnp
import unxts.linalg as ul

import unxt as u

import coordinax.charts as cxc
import coordinax.internal as cxi
import coordinax.manifolds as cxm
from coordinax.internal import (
    QuantityMatrix,
    UnitsMatrix,
    cdict_units,
    det,
    det_p,
    inv,
    inv_p,
    matmul,
    matvec,
    vecdot,
    vecmat,
)


def test_internal_reexports_unxts_linalg():
    """`coordinax.internal` re-exports the unxts.linalg types and ops."""
    assert QuantityMatrix is ul.QuantityMatrix
    assert UnitsMatrix is ul.UnitsMatrix
    assert det is ul.det
    assert det_p is ul.det_p
    assert inv is ul.inv
    assert inv_p is ul.inv_p
    assert cdict_units is ul.cdict_units
    # linalg ops (matmul/matvec/vecdot/vecmat) are re-exported at the same
    # boundary; `QMatrix` is intentionally *not* re-exported (no back-compat).
    assert matmul is ul.matmul
    assert matvec is ul.matvec
    assert vecdot is ul.vecdot
    assert vecmat is ul.vecmat
    assert not hasattr(cxi, "QMatrix")


def test_norm_uses_unxts_linalg_metric():
    """The metric/norm path works end-to-end through unxts.linalg."""
    metric = cxm.RoundMetric(ndim=2)
    at = {"theta": u.Angle(jnp.pi / 2, "rad"), "phi": u.Angle(0.0, "rad")}
    v = {"theta": u.Q(1.0, "rad/s"), "phi": u.Q(1.0, "rad/s")}
    result = cxm.norm(v, metric, cxc.sph2, at=at)
    assert jnp.allclose(u.ustrip("rad/s", result), jnp.sqrt(2.0), atol=1e-6)
