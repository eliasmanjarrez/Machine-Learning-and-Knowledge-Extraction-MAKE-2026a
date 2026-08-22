# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Step 2. The two controls that decide whether the structure is real.

Control one, circular-shift surrogates. Each joint-angle series is shifted by
a random offset before the matrix is recomputed. The shift destroys temporal
correspondence between joints while preserving each signal's own spectrum
exactly, so any excess of the observed matrix over this null cannot be
explained by the marginal spectra alone. The deposit ships the null matrices
averaged over 100 repetitions per participant; this script reports the excess
by category and, with ``--recompute``, regenerates the null from the angle
series.

Control two, shared keypoints. Angles built from overlapping keypoints inherit
correlated tracking error, which on its own would produce exactly the
within-limb structure the article reports. Pairs are therefore stratified by
how many keypoints their defining triplets have in common, and the category
comparison is repeated among the pairs that share none.

The second control is the one that carries the article's anatomical claim. A
circular shift destroys temporally aligned tracking error as well as
biological correspondence, so the surrogate excess alone cannot separate
genuine intra-limb co-modulation from shared-keypoint error. Intra-limb pairs
share at least two keypoints by construction and so cannot appear in the
zero-shared stratum at all; what survives there is the elevation of bilateral
homologous pairs over cross-limb and head-body pairs.

    python step2_surrogate_and_landmark_controls.py [--recompute] [--quick]
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from common import DT, PUBLISHED, banner, compare, load, save_json
from intrabody import categories as cat
from intrabody.wavelet import cross_wavelet, morlet_cwt

N_SURROGATE = 100


def surrogate_null(angles, n_valid, n_rep, rng):
    """Mean null matrix from circular shifts of each angle series."""
    x = angles[:, :n_valid].astype(float)
    total = np.zeros(78)
    for _ in range(n_rep):
        shifted = np.vstack([np.roll(row, int(rng.integers(n_valid)))
                             for row in x])
        cwts = [morlet_cwt(shifted[j], DT) for j in range(cat.N_ANGLES)]
        k = 0
        for i in range(cat.N_ANGLES):
            for j in range(i + 1, cat.N_ANGLES):
                total[k] += np.nanmean(
                    np.asarray(cross_wavelet(cwts[i], cwts[j])["Xp"]))
                k += 1
    return total / n_rep


def main():
    banner("Step 2  Surrogate and shared-landmark controls")
    idx = cat.category_indices()
    iu = np.triu_indices(cat.N_ANGLES, 1)
    out = {}

    # ---------------------------------------------------------- surrogates
    # The observed matrices live in comodulation_matrices.npz and are not
    # repeated in the surrogate file: one array, one place.
    sur = load("surrogate_null_matrices.npz")
    observed = np.asarray(load("comodulation_matrices.npz")["xwt"],
                          float)[:, iu[0], iu[1]]
    null = np.asarray(sur["surrogate_null"], float)[:, iu[0], iu[1]]

    if "--recompute" in sys.argv:
        angles = load("joint_angles_15hz.npz")
        A, n_valid = angles["angles"], angles["n_samples"]
        n = 5 if "--quick" in sys.argv else A.shape[0]
        reps = 5 if "--quick" in sys.argv else N_SURROGATE
        rng = np.random.default_rng(0)
        print("  regenerating the null: %d participants x %d shifts"
              % (n, reps))
        redone = np.vstack([surrogate_null(A[s], int(n_valid[s]), reps, rng)
                            for s in range(n)])
        rel = np.abs(redone.mean(0) - null[:n].mean(0)) / np.maximum(
            np.abs(null[:n].mean(0)), 1e-12)
        print("  agreement with the deposited null: median %.3f, max %.3f"
              % (float(np.median(rel)), float(rel.max())))
        print("  (a surrogate null is stochastic, so small differences are"
              " expected)")

    print("\n  excess of the observed matrix over the circular-shift null")
    excess = {}
    for c in cat.CATEGORIES:
        obs = observed[:, idx[c]].mean(axis=1)
        nul = null[:, idx[c]].mean(axis=1)
        d = obs - nul
        w, p = stats.wilcoxon(obs, nul)
        excess[c] = {"excess": float(d.mean()),
                     "cohens_d": float(d.mean() / d.std(ddof=1)),
                     "wilcoxon_p": float(p)}
        print("    %-12s +%8.1f   d = %+.2f   p = %.2e"
              % (c, d.mean(), excess[c]["cohens_d"], p))
    out["surrogate_excess"] = excess
    print("    overall excess %+.1f power units"
          % (observed - null).mean())

    # --------------------------------------------------- shared landmarks
    table = cat.pair_table()
    shared = np.array([row["shared_keypoints"] for row in table])
    values = observed

    print("\n  stratification by the number of shared keypoints")
    strata = {}
    for k in sorted(set(shared)):
        sel = np.where(shared == k)[0]
        m = float(values[:, sel].mean())
        strata[int(k)] = {"n_pairs": int(len(sel)), "mean": m}
        print("    %d shared  %2d pairs   mean %8.1f" % (k, len(sel), m))
    out["strata"] = strata

    zero = np.where(shared == 0)[0]
    present = [c for c in cat.CATEGORIES
               if any(table[k]["category"] == c for k in zero)]
    columns = [values[:, [k for k in zero if table[k]["category"] == c]].mean(
        axis=1) for c in present]
    chi, p = stats.friedmanchisquare(*columns)
    print("\n  category comparison inside the zero-shared stratum")
    for c, col in zip(present, columns):
        print("    %-12s %8.1f" % (c, col.mean()))
    print("    Friedman chi2(%d) = %.1f, p = %.2e" % (len(present) - 1, chi, p))
    print("    Intra-limb pairs cannot appear here: by construction they")
    print("    share at least two keypoints.")
    out["zero_shared_stratum"] = {
        "categories": present,
        "means": {c: float(col.mean()) for c, col in zip(present, columns)},
        "friedman_chi2": float(chi), "friedman_p": float(p)}

    print()
    compare("surrogate excess intra", excess["Intra-limb"]["excess"],
            PUBLISHED["surrogate_excess_intra"], 0.05)
    compare("surrogate excess homolog.", excess["Homologous"]["excess"],
            PUBLISHED["surrogate_excess_homologous"], 0.05)
    compare("pairs sharing three", strata[3]["mean"],
            PUBLISHED["shared_three_mean"])
    compare("pairs sharing none", strata[0]["mean"],
            PUBLISHED["shared_zero_mean"])
    compare("zero stratum homologous",
            out["zero_shared_stratum"]["means"]["Homologous"],
            PUBLISHED["zero_stratum_homologous"])
    compare("zero stratum cross-limb",
            out["zero_shared_stratum"]["means"]["Cross-limb"],
            PUBLISHED["zero_stratum_cross"])

    save_json("step2_controls.json", out)


if __name__ == "__main__":
    main()
