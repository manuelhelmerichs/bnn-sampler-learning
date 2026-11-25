"""Convergence diagnostics of chains shaped ``(chain, draw, ...)``.

There is one definition of each, following Vehtari et al. (2021): split R-hat,
rank-normalized with its folded variant, and multichain bulk and tail ESS. ESS
pools the chains: the between-chain variance enters the autocorrelation, so
chains that disagree have few effective draws however well each one mixes.
Every statistic is per coordinate; the trailing axes are kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _coordinate_batches(values):
    """Keep float64/FFT scratch bounded instead of promoting a whole trajectory."""
    if values.ndim < 2:
        raise ValueError("chains need chain and draw axes")
    flat = values.reshape(*values.shape[:2], -1)
    width = max(
        1, min(256, (16 * 1024**2) // max(1, values.shape[0] * values.shape[1] * 8 * 8))
    )
    for start in range(0, flat.shape[-1], width):
        selection = slice(start, start + width)
        yield selection, np.asarray(flat[..., selection], dtype=np.float64)


def split_rhat(chains: np.ndarray, *, split: bool = True) -> np.ndarray:
    """Split R-hat over arrays shaped ``(chain, draw, ...)``.

    With ``split=False`` each chain is used whole. Constancy is judged against the
    coordinate's float32 quantization scale: ulp-level variation (for example of
    canonicalized coordinates) reports 1, materially separated constant chains
    report ``inf``.
    """
    values = np.asarray(chains)
    if values.ndim < 2 or values.shape[0] < 2 or values.shape[1] < (4 if split else 2):
        raise ValueError("split R-hat needs at least two chains and four draws")
    output = np.empty(int(np.prod(values.shape[2:])), dtype=np.float64)
    for selection, block in _coordinate_batches(values):
        if split:
            half = block.shape[1] // 2
            block = np.concatenate((block[:, :half], block[:, -half:]), axis=0)
        n = block.shape[1]
        within = np.mean(np.var(block, axis=1, ddof=1), axis=0)
        between = n * np.var(np.mean(block, axis=1), axis=0, ddof=1)
        variance = ((n - 1) * within + between) / n
        scale = np.mean(np.abs(block), axis=(0, 1))
        quantum = np.maximum((np.finfo(np.float32).eps * scale) ** 2, 1e-30)
        rhat = np.ones_like(within)
        active = within > quantum
        rhat[active] = np.sqrt(variance[active] / within[active])
        rhat[~active & (between > quantum)] = np.inf
        output[selection] = rhat
    return output.reshape(values.shape[2:])


def rank_normalize(chains: np.ndarray) -> np.ndarray:
    """Rank-normalize each coordinate across all chain/draw observations."""
    values = np.asarray(chains)
    if values.ndim < 2:
        raise ValueError("chains need chain and draw axes")
    flat = values.reshape(values.shape[0] * values.shape[1], -1)
    ranks = rankdata(flat, axis=0, method="average")
    normalized = ndtri((ranks - 0.375) / (flat.shape[0] + 0.25))
    return normalized.reshape(values.shape)


def rank_normalized_split_rhat(
    chains: np.ndarray, *, split: bool = True
) -> dict[str, np.ndarray]:
    """Bulk, folded and combined rank-normalized R-hat (Vehtari et al., 2021).

    ``folded`` folds each coordinate around its pooled median before rank
    normalization, which detects scale differences that ``bulk`` can miss.
    """
    values = np.asarray(chains)
    bulk = np.empty(int(np.prod(values.shape[2:])), dtype=np.float64)
    folded = np.empty_like(bulk)
    for selection, block in _coordinate_batches(values):
        bulk[selection] = split_rhat(rank_normalize(block), split=split)
        median = np.median(block, axis=(0, 1), keepdims=True)
        folded[selection] = split_rhat(
            rank_normalize(np.abs(block - median)), split=split
        )
    bulk = bulk.reshape(values.shape[2:])
    folded = folded.reshape(values.shape[2:])
    return {"bulk": bulk, "folded": folded, "max": np.maximum(bulk, folded)}


def _draws(chains: np.ndarray) -> np.ndarray:
    values = np.asarray(chains)
    if values.ndim < 2 or values.shape[1] < 4:
        raise ValueError("ESS needs chain and draw axes and at least four draws")
    return values


def _multichain_ess(block: np.ndarray) -> np.ndarray:
    """ESS of float64 ``(chain, draw, coordinate)`` values, each chain split in two.

    Geyer's initial monotone sequence truncates the autocorrelations. A coordinate
    with any constant chain has zero ESS: an unobserved transition provides no
    evidence of effective sampling, for raw values, ranks and tail indicators.
    """
    moving = np.any(block != block[:, :1], axis=1).all(axis=0)
    half = block.shape[1] // 2
    block = np.concatenate((block[:, :half], block[:, -half:]), axis=0)
    chains, draws, _ = block.shape
    centered = block - block.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, n=2 * draws, axis=1)
    autocovariance = (
        np.fft.irfft(spectrum * spectrum.conj(), n=2 * draws, axis=1)[:, :draws] / draws
    )
    within = np.mean(np.var(block, axis=1, ddof=1), axis=0)
    variance = (draws - 1) / draws * within + np.var(block.mean(axis=1), axis=0, ddof=1)
    active = moving & (variance > 0) & np.isfinite(variance)
    rho = np.zeros(autocovariance.shape[1:])
    np.divide(within - autocovariance.mean(axis=0), variance, out=rho, where=active)
    rho = 1 - rho
    rho[0] = 1
    pairs = rho[: draws // 2 * 2].reshape(draws // 2, 2, -1).sum(axis=1)
    positive = np.logical_and.accumulate(pairs > 0, axis=0)
    monotone = np.minimum.accumulate(pairs, axis=0)
    tau = np.maximum(1.0, -1 + 2 * np.sum(monotone, axis=0, where=positive))
    total = chains * draws
    return np.where(active, np.minimum(total / tau, total), 0.0)


def multichain_ess(chains: np.ndarray) -> np.ndarray:
    """ESS of the values themselves, for the precision of their mean."""
    values = _draws(chains)
    output = np.empty(int(np.prod(values.shape[2:])), dtype=np.float64)
    for selection, block in _coordinate_batches(values):
        output[selection] = _multichain_ess(block)
    return output.reshape(values.shape[2:])


def bulk_tail_ess(chains: np.ndarray) -> dict[str, np.ndarray]:
    """Bulk ESS of the ranks, and tail ESS of the 5% and 95% quantiles.

    Tail ESS is the smaller ESS of the two quantile indicators.
    """
    values = _draws(chains)
    bulk = np.empty(int(np.prod(values.shape[2:])), dtype=np.float64)
    tail = np.empty_like(bulk)
    for selection, block in _coordinate_batches(values):
        bulk[selection] = _multichain_ess(rank_normalize(block))
        lower, upper = np.quantile(block, (0.05, 0.95), axis=(0, 1))
        tail[selection] = np.minimum(
            _multichain_ess((block <= lower).astype(np.float64)),
            _multichain_ess((block >= upper).astype(np.float64)),
        )
    shape = values.shape[2:]
    return {"bulk": bulk.reshape(shape), "tail": tail.reshape(shape)}


def chain_diagnostics(chains: np.ndarray) -> dict[str, Any]:
    """R-hat, bulk and tail ESS of every coordinate and their worst values, as JSON."""
    rhat = rank_normalized_split_rhat(chains)["max"].reshape(-1)
    ess = bulk_tail_ess(chains)
    bulk, tail = ess["bulk"].reshape(-1), ess["tail"].reshape(-1)
    return {
        "rhat": rhat.tolist(),
        "bulk_ess": bulk.tolist(),
        "tail_ess": tail.tolist(),
        "rhat_max": float(rhat.max()),
        "bulk_ess_min": float(bulk.min()),
        "tail_ess_min": float(tail.min()),
    }
