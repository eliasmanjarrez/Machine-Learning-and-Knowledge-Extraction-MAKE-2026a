# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Morlet wavelet core: CWT, cross-wavelet transform and wavelet coherence.

This is the exact implementation used to produce every time-frequency matrix
reported in the article. It follows the Morlet formulation of Torrence and
Compo (1998) with a central frequency of omega_0 = 6, and the coherence
follows Grinsted et al. (2004).

The three entry points are:

    morlet_cwt(signal, dt)        continuous wavelet transform of one signal
    cross_wavelet(cwt_a, cwt_b)   cross-wavelet transform of a pair
    wavelet_coherence(cwt_a, cwt_b)   normalized counterpart of the above

A note on what the cross-wavelet modulus measures, because the article turns
on it: since |W_a conj(W_b)| = |W_a| |W_b|, the modulus is the product of the
two marginal amplitudes with the phase difference discarded. It quantifies
simultaneous amplitude, not phase locking. Perfect phase locking and
uniformly random phase give the same value. The article therefore calls the
quantity amplitude co-modulation, and reports wavelet coherence separately as
the normalized control.

References
----------
Torrence, C.; Compo, G.P. A practical guide to wavelet analysis.
    Bull. Am. Meteorol. Soc. 1998, 79, 61-78.
Grinsted, A.; Moore, J.C.; Jevrejeva, S. Application of the cross wavelet
    transform and wavelet coherence to geophysical time series.
    Nonlinear Process. Geophys. 2004, 11, 561-566.
"""
from __future__ import annotations

import numpy as np

OMEGA0 = 6.0
N_SCALES = 48
MAX_PERIOD_S = 60.0


def morlet_cwt(signal, dt, n_scales=N_SCALES, omega0=OMEGA0,
               max_period=MAX_PERIOD_S):
    """Continuous wavelet transform with a Morlet mother wavelet.

    Parameters
    ----------
    signal : array_like, shape (n,)
        Time series. Missing values are forward-filled before the transform,
        which is what the pipeline does; the signal is then demeaned.
    dt : float
        Sampling interval in seconds. The article uses 4/60 s for every
        population-level analysis and 1/60 s for single-session displays.
    n_scales : int
        Number of logarithmically spaced scales.
    omega0 : float
        Morlet central frequency.
    max_period : float
        Longest period considered, in seconds.

    Returns
    -------
    dict with keys
        Wr, Wi : list of ndarray
            Real and imaginary parts of the coefficients, one array per scale.
        scales, freqs : ndarray
            Scales in seconds and their equivalent frequencies in Hz, using
            f = omega0 / (2 pi s).
        coi : ndarray
            Cone of influence, in seconds from the nearest edge.
        power : ndarray, shape (n_scales, n)
            Squared modulus of the coefficients.
        n, dt : int, float

    Notes
    -----
    The transform is computed in the Fourier domain. The signal is zero-padded
    to the next power of two to limit edge effects, and only the first `n`
    samples of the inverse transform are kept.
    """
    x = np.asarray(signal, dtype=np.float64).copy()

    # Forward-fill missing values, then demean. Both steps match the pipeline.
    nan_mask = np.isnan(x)
    if nan_mask.any():
        if nan_mask.all():
            raise ValueError("signal is entirely missing")
        for i in range(1, len(x)):
            if nan_mask[i]:
                x[i] = x[i - 1]
        first_valid = int(np.argmin(nan_mask))
        x[:first_valid] = x[first_valid]
    x = x - x.mean()

    n = len(x)

    n_fft = 1
    while n_fft < n:
        n_fft <<= 1
    padded = np.zeros(n_fft)
    padded[:n] = x
    spectrum = np.fft.fft(padded)

    # Angular frequencies, positive in the first half and negative in the
    # second, as required by the analytic wavelet below.
    k = np.arange(n_fft)
    omega = np.where(k < n_fft // 2,
                     2.0 * np.pi * k / (n_fft * dt),
                     -2.0 * np.pi * (n_fft - k) / (n_fft * dt))

    s0 = 2.0 * dt
    s_max = min(n * dt / 4.0, max_period)
    dj = np.log2(s_max / s0) / (n_scales - 1)
    scales = s0 * 2.0 ** (np.arange(n_scales) * dj)
    freqs = omega0 / (2.0 * np.pi * scales)

    Wr, Wi = [], []
    power = np.zeros((n_scales, n))
    for j, s in enumerate(scales):
        norm = np.pi ** -0.25 * np.sqrt(2.0 * np.pi * s / dt)
        # Analytic wavelet: only positive frequencies contribute.
        psi = np.where(omega > 0,
                       norm * np.exp(-0.5 * (s * omega - omega0) ** 2), 0.0)
        coeffs = np.fft.ifft(spectrum * psi)
        wr = coeffs[:n].real.copy()
        wi = coeffs[:n].imag.copy()
        Wr.append(wr)
        Wi.append(wi)
        power[j] = wr * wr + wi * wi

    t = np.arange(n)
    coi = np.minimum(t, n - 1 - t) * dt / np.sqrt(2.0)

    return {"Wr": Wr, "Wi": Wi, "scales": scales, "freqs": freqs,
            "coi": coi, "power": power, "n": n, "dt": dt}


def cross_wavelet(cwt_a, cwt_b):
    """Cross-wavelet transform of two continuous wavelet transforms.

    Computes conj(W_a) W_b scale by scale. The returned ``Xp`` is the modulus
    that the article reports as amplitude co-modulation.

    Returns
    -------
    dict with keys Xr, Xi, Xp, each a list of arrays, one per scale.
    """
    Xr, Xi, Xp = [], [], []
    for ar, ai, br, bi in zip(cwt_a["Wr"], cwt_a["Wi"],
                              cwt_b["Wr"], cwt_b["Wi"]):
        xr = ar * br + ai * bi
        xi = ai * br - ar * bi
        Xr.append(xr)
        Xi.append(xi)
        Xp.append(np.sqrt(xr * xr + xi * xi))
    return {"Xr": Xr, "Xi": Xi, "Xp": Xp}


def _smooth(real_parts, imag_parts, scales, dt):
    """Gaussian smoothing in time, one kernel width per scale.

    Coherence requires smoothing, otherwise it is identically one everywhere.
    The kernel half-width is half a scale expressed in samples, which is the
    convention used throughout the pipeline.
    """
    out_r, out_i = [], []
    for j, s in enumerate(scales):
        hw = max(1, int(round(s / dt * 0.5)))
        offsets = np.arange(-hw, hw + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / max(hw, 1)) ** 2)
        kernel /= kernel.sum()
        out_r.append(np.convolve(real_parts[j], kernel, mode="same"))
        out_i.append(np.convolve(imag_parts[j], kernel, mode="same"))
    return out_r, out_i


def wavelet_coherence(cwt_a, cwt_b):
    """Wavelet coherence, the normalized counterpart of the cross-wavelet.

    Defined as the smoothed cross-spectrum divided by the product of the
    smoothed individual spectra, which bounds the result between 0 and 1 and
    removes the influence of marginal power.

    Returns
    -------
    dict with keys
        coh : ndarray, shape (n_scales, n)
        phase_r, phase_i : ndarray
            Real and imaginary parts of the smoothed cross-spectrum. Note that
            ``coh`` alone carries no information about the phase angle.
        scales, freqs, coi : ndarray
    """
    scales = cwt_a["scales"]
    dt = cwt_a["dt"]
    n_scales = len(scales)
    n = cwt_a["n"]

    p_a = [ar * ar + ai * ai for ar, ai in zip(cwt_a["Wr"], cwt_a["Wi"])]
    p_b = [br * br + bi * bi for br, bi in zip(cwt_b["Wr"], cwt_b["Wi"])]
    xwt = cross_wavelet(cwt_a, cwt_b)

    zeros = [np.zeros(n) for _ in range(n_scales)]
    sa_r, _ = _smooth(p_a, zeros, scales, dt)
    sb_r, _ = _smooth(p_b, zeros, scales, dt)
    sx_r, sx_i = _smooth(xwt["Xr"], xwt["Xi"], scales, dt)

    coh = np.zeros((n_scales, n))
    phase_r = np.zeros((n_scales, n))
    phase_i = np.zeros((n_scales, n))
    for j in range(n_scales):
        denom = sa_r[j] * sb_r[j]
        numer = sx_r[j] ** 2 + sx_i[j] ** 2
        ok = denom > 1e-30
        coh[j, ok] = np.minimum(1.0, numer[ok] / denom[ok])
        phase_r[j] = sx_r[j]
        phase_i[j] = sx_i[j]

    return {"coh": coh, "phase_r": phase_r, "phase_i": phase_i,
            "scales": scales, "freqs": cwt_a["freqs"], "coi": cwt_a["coi"]}


def coi_mask(cwt):
    """Boolean mask of the time-frequency points inside the cone of influence.

    A point is inside when it is further from either edge of the record than
    the e-folding time of the wavelet at that scale. The cone is what the
    single-session displays shade in Figure 1d and Figure 5a of the article.
    """
    n_scales = len(cwt["scales"])
    mask = np.zeros((n_scales, cwt["n"]), dtype=bool)
    for j, s in enumerate(cwt["scales"]):
        mask[j] = cwt["coi"] >= s
    return mask


def session_mean_modulus(cwt_a, cwt_b, restrict_to_coi=False):
    """Session-averaged cross-wavelet modulus for one pair of signals.

    This is the single number that fills one off-diagonal cell of the 13 x 13
    matrices distributed with this deposit.

    Parameters
    ----------
    restrict_to_coi : bool
        With the default ``False`` the average runs over the whole
        time-frequency plane, which is how every matrix in the article was
        produced and what ``comodulation_matrices.npz`` contains. Setting it
        to ``True`` averages only inside the cone of influence of both
        signals.

    Notes
    -----
    The two options are not interchangeable. Restricting to the cone drops the
    edges of the slowest scales, where the modulus is largest, so it lowers
    every cell by roughly a tenth. The article reports the unrestricted
    average; ``step1_comodulation_matrices.py`` recomputes both and shows that
    the anatomical ordering, the omnibus test and every pairwise comparison
    are the same either way, which is why the choice does not affect any
    conclusion drawn in the paper.
    """
    xwt = cross_wavelet(cwt_a, cwt_b)
    stack = np.asarray(xwt["Xp"])
    if not restrict_to_coi:
        return float(np.nanmean(stack))
    mask = coi_mask(cwt_a) & coi_mask(cwt_b)
    values = stack[mask]
    return float(np.nanmean(values)) if values.size else float("nan")
