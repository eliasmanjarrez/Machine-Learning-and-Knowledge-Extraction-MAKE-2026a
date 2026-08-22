# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Step 5. Within-session temporal consistency, with the participant as the unit.

A population mean can be perfectly flat while every individual reorganizes
underneath it, so a flat group trend says nothing about whether any one person
is consistent. This step tests consistency inside each participant.

The unit of inference matters here more than anywhere else in the article. A
first version of this comparison pooled every within-person and between-person
correlation into one rank test, which entered each participant thousands of
times and produced a p-value that could not be read as evidence about people.
The comparison below collapses each participant to a single pair of numbers
first, the Fisher-z means of their within-person and between-person
correlations, and then tests those 70 pairs.

Two further numbers keep the result in proportion. An equivalence test against
a pre-stated bound of five per cent total change across the session does not
succeed, and the mean intraclass correlation across the 78 pairs is about 0.3
with no single pair above 0.5. Whatever is consistent lives in the pattern
across pairs, not in any pair on its own, and the evidence supports
within-session consistency rather than a stable trait.

    python step5_temporal_consistency.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from common import PUBLISHED, banner, compare, load, save_json
from intrabody import categories as cat
from intrabody.statistics import fisher_z

N_PERMUTATIONS = 10000


def main():
    banner("Step 5  Within-session temporal consistency")
    w = np.asarray(load("window_profiles.npz")["xwt"], float)
    n, n_windows = w.shape[0], w.shape[1]
    iu = np.triu_indices(cat.N_ANGLES, 1)
    profiles = np.log10(np.maximum(w, 1e-6))[:, :, iu[0], iu[1]]
    print("  %d participants, %d windows of 2 minutes" % (n, n_windows))

    flat = profiles.reshape(n * n_windows, -1)
    centred = flat - flat.mean(1, keepdims=True)
    centred /= np.sqrt((centred ** 2).sum(1, keepdims=True)) + 1e-30
    corr = np.einsum("ik,jk->ij", centred, centred)
    owner = np.repeat(np.arange(n), n_windows)

    within_z = np.zeros(n)
    between_z = np.zeros(n)
    tri = np.triu_indices(n_windows, 1)
    for s in range(n):
        rows = np.where(owner == s)[0]
        others = np.where(owner != s)[0]
        within_z[s] = fisher_z(corr[np.ix_(rows, rows)][tri]).mean()
        between_z[s] = fisher_z(corr[np.ix_(rows, others)].ravel()).mean()

    diff = within_z - between_z
    w_stat, p_w = stats.wilcoxon(within_z, between_z)
    se = diff.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)

    print("\n  participant-level comparison")
    print("    within-person  Fisher-z %.3f   (r = %.3f)"
          % (within_z.mean(), np.tanh(within_z.mean())))
    print("    between-person Fisher-z %.3f   (r = %.3f)"
          % (between_z.mean(), np.tanh(between_z.mean())))
    print("    difference %.3f, 95%% CI %.3f to %.3f"
          % (diff.mean(), diff.mean() - crit * se, diff.mean() + crit * se))
    print("    Wilcoxon signed-rank W = %.0f, p = %.2e, paired d = %.2f"
          % (w_stat, p_w, diff.mean() / diff.std(ddof=1)))
    print("    participants with within above between: %d of %d"
          % (int((diff > 0).sum()), n))

    # Permutation that keeps everything except who each window belongs to.
    rng = np.random.default_rng(0)
    z_all = fisher_z(corr)
    np.fill_diagonal(z_all, 0.0)
    row_sums = z_all.sum(axis=1)
    n_within_cells = n_windows * (n_windows - 1)
    n_between_cells = n_windows * (n * n_windows - n_windows)
    null = np.empty(N_PERMUTATIONS)
    index = np.arange(n * n_windows)
    for it in range(N_PERMUTATIONS):
        groups = rng.permutation(index).reshape(n, n_windows)
        blocks = z_all[groups[:, :, None], groups[:, None, :]]
        inside = blocks.sum(axis=(1, 2))
        outside = row_sums[groups].sum(axis=1) - inside
        null[it] = (inside / n_within_cells - outside / n_between_cells).mean()
    p_perm = (1.0 + np.sum(null >= diff.mean())) / (N_PERMUTATIONS + 1.0)
    print("    permutation null %.4f +/- %.4f, p %s"
          % (null.mean(), null.std(ddof=1),
             "< 1e-4" if p_perm < 1e-4 else "%.4f" % p_perm))

    # Reliability of a single pair, which is poor and belongs in the record.
    icc = []
    for k in range(profiles.shape[2]):
        x = profiles[:, :, k]
        grand = x.mean()
        msb = n_windows * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
        msw = ((x - x.mean(axis=1, keepdims=True)) ** 2).sum() / (
            n * (n_windows - 1))
        icc.append((msb - msw) / (msb + (n_windows - 1) * msw))
    icc = np.array(icc)
    print("\n  reliability of individual pairs")
    print("    mean intraclass correlation %.3f, range %.3f to %.3f"
          % (icc.mean(), icc.min(), icc.max()))
    print("    pairs above 0.5: %d of %d" % (int((icc > 0.5).sum()), len(icc)))

    # Global trend, and the equivalence test that does not succeed.
    #
    # The two tests are taken on slightly different summaries of the same
    # windows, which is how they were run for the article and is reproduced
    # here rather than silently harmonized. The trend is the slope of overall
    # power, the mean of the 78 off-diagonal pairs in the original units. The
    # equivalence test needs a relative bound, so it works on log10 power
    # averaged over the matrix. Both are null results and the choice does not
    # affect either conclusion: every variant we tried leaves the trend
    # non-significant and equivalence unestablished.
    def slopes_of(series):
        t = np.arange(n_windows, dtype=float)
        tc = t - t.mean()
        return np.array([float((tc * (g - g.mean())).sum() / (tc ** 2).sum())
                         for g in series])

    overall = w[:, :, iu[0], iu[1]].mean(axis=2)
    slopes = slopes_of(overall)
    _, p_slope = stats.wilcoxon(slopes)

    log_power = np.log10(np.maximum(w, 1e-6)).mean(axis=(2, 3))
    total = slopes_of(log_power) * (n_windows - 1)
    bound = np.log10(1.05)                       # a five per cent bound
    m = total.mean()
    se_t = total.std(ddof=1) / np.sqrt(len(total))
    p_tost = max(stats.t.sf((m + bound) / se_t, len(total) - 1),
                 stats.t.cdf((m - bound) / se_t, len(total) - 1))
    print("\n  global trend across the session")
    print("    positive slopes in %.1f%% of participants, Wilcoxon p = %.2f"
          % (100 * np.mean(slopes > 0), p_slope))
    print("    mean total change %.2f%%, equivalence test p = %.2f (%s)"
          % (100 * (10 ** m - 1), p_tost,
             "equivalent" if p_tost < 0.05 else "not established"))

    out = {"within_z": float(within_z.mean()),
           "between_z": float(between_z.mean()),
           "difference": float(diff.mean()),
           "ci95": [float(diff.mean() - crit * se),
                    float(diff.mean() + crit * se)],
           "wilcoxon_p": float(p_w),
           "paired_d": float(diff.mean() / diff.std(ddof=1)),
           "n_within_above_between": int((diff > 0).sum()),
           "permutation_p": float(p_perm),
           "icc_mean": float(icc.mean()),
           "icc_above_0p5": int((icc > 0.5).sum()),
           "slope_p": float(p_slope),
           "equivalence_p": float(p_tost)}

    print()
    compare("within-person Fisher-z", out["within_z"],
            PUBLISHED["consistency_within_z"])
    compare("between-person Fisher-z", out["between_z"],
            PUBLISHED["consistency_between_z"])
    compare("mean intraclass corr.", out["icc_mean"], PUBLISHED["icc_mean"])

    save_json("step5_temporal_consistency.json", out)


if __name__ == "__main__":
    main()
