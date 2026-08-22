# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Step 3. Wavelet coherence as the normalized control.

The cross-wavelet modulus is a product of two marginal amplitudes, so joints
that simply move a lot score high. Wavelet coherence divides that influence
out by construction. If the anatomical structure were an artefact of using an
unnormalized statistic, coherence would not show it. It does, with a stronger
omnibus test.

The article nevertheless keeps amplitude co-modulation as its primary measure,
and the reason is visible in the numbers this script prints. Coherence
saturates for pairs of joints within one body: all four categories fall in a
narrow band, and two participants' coherence profiles are nearly identical.
That leaves the measure almost no range in which to express individual
differences, which is what the identification analysis needs.

One caveat is worth stating because it is easy to overstate the result.
Coherence magnitude says nothing about the phase angle. That profiles agree
across people means the magnitudes agree; it does not mean the phase relations
themselves are uniform. Establishing that would require phase-angle statistics,
which are not computed here or in the article.

    python step3_coherence_control.py [--recompute] [--quick]
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from common import DT, PUBLISHED, banner, compare, load, save_json
from intrabody import categories as cat
from intrabody.statistics import zscore_rows
from intrabody.wavelet import morlet_cwt, wavelet_coherence


def coherence_matrix(angles, n_valid):
    """The 78 coherence values of one participant."""
    x = angles[:, :n_valid].astype(float)
    cwts = [morlet_cwt(x[j], DT) for j in range(cat.N_ANGLES)]
    out = []
    for i in range(cat.N_ANGLES):
        for j in range(i + 1, cat.N_ANGLES):
            out.append(float(np.nanmean(
                wavelet_coherence(cwts[i], cwts[j])["coh"])))
    return np.array(out)


def main():
    banner("Step 3  Wavelet coherence control")
    iu = np.triu_indices(cat.N_ANGLES, 1)
    idx = cat.category_indices()

    wtc = np.asarray(load("coherence_matrices.npz")["wtc"], float)[
        :, iu[0], iu[1]]
    xwt = np.asarray(load("comodulation_matrices.npz")["xwt"], float)[
        :, iu[0], iu[1]]

    if "--recompute" in sys.argv:
        angles = load("joint_angles_15hz.npz")
        A, n_valid = angles["angles"], angles["n_samples"]
        n = 5 if "--quick" in sys.argv else A.shape[0]
        print("  recomputing coherence for %d participants" % n)
        redone = np.vstack([coherence_matrix(A[s], int(n_valid[s]))
                            for s in range(n)])
        rel = np.abs(redone - wtc[:n]) / np.maximum(np.abs(wtc[:n]), 1e-12)
        print("  agreement with the deposit: max %.2e" % rel.max())

    print("\n  category means")
    columns = [wtc[:, idx[c]].mean(axis=1) for c in cat.CATEGORIES]
    for c, col in zip(cat.CATEGORIES, columns):
        print("    %-12s %.3f" % (c, col.mean()))
    chi, p = stats.friedmanchisquare(*columns)
    chi_x, p_x = stats.friedmanchisquare(
        *[xwt[:, idx[c]].mean(axis=1) for c in cat.CATEGORIES])
    print("    Friedman chi2 = %.1f, p = %.2e   (modulus: %.1f, %.2e)"
          % (chi, p, chi_x, p_x))

    # Why the modulus remains the primary measure.
    def spread(m):
        col = np.array([m[:, idx[c]].mean(axis=1).mean()
                        for c in cat.CATEGORIES])
        return 100 * (col.max() - col.min()) / col.mean()

    def between_subject_cv(m):
        per = m.mean(axis=1)
        return 100 * per.std(ddof=1) / per.mean()

    def profile_similarity(m):
        z = zscore_rows(m)
        c = np.einsum("ik,jk->ij", z, z) / z.shape[1]
        off = ~np.eye(len(m), dtype=bool)
        return float(c[off].mean())

    print("\n  range available to each measure")
    print("    spread across categories   coherence %5.1f%%   modulus %5.1f%%"
          % (spread(wtc), spread(xwt)))
    print("    between-subject CV         coherence %5.2f%%   modulus %5.1f%%"
          % (between_subject_cv(wtc), between_subject_cv(xwt)))
    print("    mean profile correlation   coherence %5.2f    modulus %5.2f"
          % (profile_similarity(wtc), profile_similarity(xwt)))
    print("    Coherence magnitude is nearly the same in every participant,")
    print("    which is why it cannot carry the identification analysis.")

    out = {
        "category_means": {c: float(col.mean())
                           for c, col in zip(cat.CATEGORIES, columns)},
        "friedman_chi2": float(chi), "friedman_p": float(p),
        "modulus_friedman_chi2": float(chi_x),
        "spread_pct": {"coherence": float(spread(wtc)),
                       "modulus": float(spread(xwt))},
        "between_subject_cv_pct": {"coherence": float(between_subject_cv(wtc)),
                                   "modulus": float(between_subject_cv(xwt))},
        "mean_profile_correlation": {
            "coherence": profile_similarity(wtc),
            "modulus": profile_similarity(xwt)},
    }

    print()
    for c, key in zip(cat.CATEGORIES,
                      ["coherence_intra", "coherence_homologous",
                       "coherence_cross", "coherence_head"]):
        compare("coherence %s" % c, out["category_means"][c], PUBLISHED[key])

    save_json("step3_coherence.json", out)


if __name__ == "__main__":
    main()
