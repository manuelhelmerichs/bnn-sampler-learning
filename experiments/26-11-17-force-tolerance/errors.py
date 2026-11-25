"""Injected force errors of known size and type.

A chain moves with ``grad U + delta * e``, with ``e`` scaled so that
``E_pi |e|^2 = E_pi |grad U|^2``: delta is the RMS error relative to the RMS
force on the typical set.

- ``noise``: fresh Gaussian noise ``e = s xi`` at every evaluation.
- ``conservative``: the frozen field ``e(u) = B (u - c)`` with ``B`` symmetric,
  the gradient of the quadratic ``V = (u - c)' B (u - c) / 2``.
- ``curl``: the same with ``B`` antisymmetric, a field without potential.

``B = Q' M Q``, with ``M`` diagonal or block-antisymmetric and ``Q`` a fast
pseudo-random rotation (a permutation, then twice random signs and an
orthonormal DCT), so a field costs O(D log D). The offset ``c`` is an exact
draw. A target's fields are drawn once and scaled for every delta.
"""

import numpy as np

TYPES = ("noise", "conservative", "curl")
NAMES = ("permutation", "inverse", "first", "second", "diagonal", "rates", "offset")


def _dct(x, xp):
    if xp is np:
        from scipy.fft import dct, idct
    else:
        from jax.scipy.fft import dct, idct
    return dct, idct


def rotate(z, field, xp=np):
    dct, _ = _dct(z, xp)
    z = xp.take_along_axis(z, xp.broadcast_to(field["permutation"], z.shape), -1)
    return dct(field["second"] * dct(field["first"] * z, norm="ortho"), norm="ortho")


def rotate_back(y, field, xp=np):
    _, idct = _dct(y, xp)
    z = field["first"] * idct(field["second"] * idct(y, norm="ortho"), norm="ortho")
    return xp.take_along_axis(z, xp.broadcast_to(field["inverse"], z.shape), -1)


def mix(z, field, xp=np):
    """``M z``: the diagonal part plus 2x2 rotations of consecutive pairs."""
    pairs = field["rates"].shape[-1]
    head = z[..., : 2 * pairs].reshape(*z.shape[:-1], pairs, 2)
    turned = xp.stack(
        [field["rates"] * head[..., 1], -field["rates"] * head[..., 0]], -1
    ).reshape(*z.shape[:-1], 2 * pairs)
    tail = xp.zeros_like(z[..., 2 * pairs :])
    return field["diagonal"] * z + xp.concatenate([turned, tail], -1)


def linear(u, field, xp=np):
    """``B (u - c)``, before scaling."""
    return rotate_back(
        mix(rotate(u - field["offset"], field, xp), field, xp), field, xp
    )


def matrix(field):
    """The dense ``B`` of a field, for normalization and checks."""
    d = len(field["offset"])
    origin = dict(field, offset=np.zeros(d))
    return linear(np.eye(d), origin).T


def draw(frame, kind, rng):
    """One target's field of ``kind``, scaled to delta = 1."""
    d = frame.dimension
    permutation = rng.permutation(d)
    field = {
        "permutation": permutation,
        "inverse": np.argsort(permutation),
        "first": rng.choice([-1.0, 1.0], d),
        "second": rng.choice([-1.0, 1.0], d),
        "diagonal": rng.normal(size=d) if kind == "conservative" else np.zeros(d),
        "rates": rng.normal(size=d // 2) if kind == "curl" else np.zeros(d // 2),
        "offset": frame.sample(1, rng)[0],
    }
    if kind == "noise":
        return field, float(np.sqrt(frame.force_power / d)), 0.0
    # E |B (u - c)|^2 = tr(B S B') + |B c|^2, with the aligned mean 0 and the
    # aligned covariance S diagonal.
    b = matrix(field)
    power = np.sum(b**2 * frame.variances) + np.sum((b @ field["offset"]) ** 2)
    return field, 0.0, float(np.sqrt(frame.force_power / power))


def arms(types, deltas):
    """The exact arm, then every type at every delta."""
    return [("exact", 0.0), *((kind, delta) for kind in types for delta in deltas)]


def stack(frames, arm_list, rng_for):
    """Field parameters of every (arm, target), and their two scales.

    ``rng_for(target, kind)`` gives the generator of one target's field.
    Returns arrays with leading axes ``(arm, target)``.
    """
    d = frames[0].dimension
    shapes = {"rates": (d // 2,)}
    fields = {name: [] for name in NAMES}
    noise, scale = [], []
    for kind, delta in arm_list:
        rows = {name: [] for name in NAMES}
        for index, frame in enumerate(frames):
            if kind == "exact":
                field = {name: np.zeros(shapes.get(name, (d,))) for name in NAMES} | {
                    "permutation": np.arange(d),
                    "inverse": np.arange(d),
                }
                field["first"] = field["second"] = np.ones(d)
                s, f = 0.0, 0.0
            else:
                field, s, f = draw(frame, kind, rng_for(index, kind))
            for name in NAMES:
                rows[name].append(field[name])
            noise.append(delta * s)
            scale.append(delta * f)
        for name in NAMES:
            fields[name].append(np.stack(rows[name]))
    shape = (len(arm_list), len(frames))
    fields = {name: np.stack(value) for name, value in fields.items()}
    return (
        fields,
        np.asarray(noise).reshape(shape),
        np.asarray(scale).reshape(shape),
    )
