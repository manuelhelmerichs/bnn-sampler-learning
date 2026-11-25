"""Compute against error for repeated runs of samplers on targets with known answers.

A method runs through a ladder of settings, and every repetition records its
estimate and its counted costs at each rung. Two methods are compared at equal
RMSE: the cost a reference needs for a method's RMSE follows from interpolating
the reference's curve, and the cost of one learned-component evaluation at which
both need the same work is the break-even.
"""

from time import perf_counter

import numpy as np


def rmse(errors, axis=0):
    return np.sqrt(np.mean(np.square(errors), axis=axis))


def cost_at(costs, errors, error):
    """Cost at which a reference curve reaches ``error``, and whether that is off it.

    Interpolates log cost in log RMSE between the reference's rungs. Beyond them
    the cost follows the Monte Carlo rate, cost proportional to RMSE^-2, from the
    nearest rung.
    """
    order = np.argsort(errors)
    x, y, target = np.log(errors[order]), np.log(costs[order]), np.log(error)
    below, above = target < x[0], target > x[-1]
    value = np.where(
        below,
        y[0] - 2 * (target - x[0]),
        np.where(above, y[-1] - 2 * (target - x[-1]), np.interp(target, x, y)),
    )
    return np.exp(value), below | above


def nested_resamples(groups, count, rng):
    """Bootstrap indexes: resample the groups, then the members within each group.

    Returns an array of shape ``(count, len(groups))`` of indexes into ``groups``.
    """
    members = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    return np.stack(
        [
            np.concatenate(
                [
                    rng.choice(members[group], len(members[group]))
                    for group in rng.integers(len(members), size=len(members))
                ]
            )
            for _ in range(count)
        ]
    )


def break_even(reference_work, work, evaluations):
    """Cost of one component evaluation at which a method needs the reference's work.

    ``work`` is the method's counted cost without its component evaluations, all
    in the reference's units. Below the break-even the method is cheaper.
    """
    return (reference_work - work) / evaluations


def ladder(block, carry, key, rungs):
    """Continue a compiled block of transitions through every rung of a ladder.

    ``block(key, carry, steps)`` returns the new carry first, then its outputs.
    Yields the carry, the outputs and the seconds spent in blocks so far, once
    per rung; the caller's own work between rungs is not timed.
    """
    import jax

    previous, seconds = 0, 0.0
    for rung in rungs:
        key, subkey = jax.random.split(key)
        start = perf_counter()
        carry, *outputs = jax.block_until_ready(block(subkey, carry, rung - previous))
        seconds += perf_counter() - start
        yield carry, outputs, seconds
        previous = rung
