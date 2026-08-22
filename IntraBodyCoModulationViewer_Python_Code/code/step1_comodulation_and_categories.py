# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Step 1. Co-modulation matrices and the anatomical category comparison.

Recomputes the 13 x 13 amplitude co-modulation matrix of each participant from
the deposited joint-angle series, checks it against the matrices distributed
with this deposit, and reproduces the category statistics reported in Table 3
and Figure 2c of the article.

The script also settles a question the Methods raise. The session average runs
over the whole time-frequency plane. Restricting it to the cone of influence
is defensible and gives systematically lower values, so both are computed here
and compared: the cone lowers every category by roughly a tenth and leaves the
ordering, the omnibus test and all six pairwise comparisons unchanged.

Run time is about four minutes for the full sample on a laptop. Pass
``--quick`` to run the first ten participants only.

    python step1_comodulation_and_categories.py [--quick] [--no-coi-check]
"""
from __future__ import annotations

import sys
import time

import numpy as np
from scipy import stats

from common import DT, PUBLISHED, banner, compare, load, save_json
from intrabody import categories as cat
from intrabody.wavelet import cross_wavelet, coi_mask, morlet_cwt


def matrices_for(angles, n_valid, with_coi):
    """Both variants of one participant's 78 pair values."""
    x = angles[:, :n_valid].astype(float)
    cwts = [morlet_cwt(x[j], DT) for j in range(cat.N_ANGLES)]
    masks = [coi_mask(c) for c in cwts] if with_coi else None

    plain, coned = [], []
    for i in range(cat.N_ANGLES):
        for j in range(i + 1, cat.N_ANGLES):
            modulus = np.asarray(cross_wavelet(cwts[i], cwts[j])["Xp"])
            plain.append(float(np.nanmean(modulus)))
            if with_coi:
                m = masks[i] & masks[j]
                coned.append(float(np.nanmean(modulus[m])))
    return np.array(plain), (np.array(coned) if with_coi else None)


def category_report(values, label, out):
    """Friedman omnibus and the six pairwise comparisons, one value per subject."""
    idx = cat.category_indices()
    columns = [values[:, idx[c]].mean(axis=1) for c in cat.CATEGORIES]
    chi, p = stats.friedmanchisquare(*columns)
    order = [cat.CATEGORIES[k] for k in np.argsort([-c.mean() for c in columns])]

    print("\n  %s" % label)
    for c, col in zip(cat.CATEGORIES, columns):
        print("    %-12s mean %8.1f   sd %7.1f" % (c, col.mean(),
                                                   col.std(ddof=1)))
    print("    ordering    %s" % " > ".join(order))
    print("    Friedman    chi2 = %.1f, p = %.2e" % (chi, p))

    pairwise = {}
    for a in range(4):
        for b in range(a + 1, 4):
            w, pw = stats.wilcoxon(columns[a], columns[b])
            d = float(np.mean(columns[a] - columns[b])
                      / np.std(columns[a] - columns[b], ddof=1))
            key = "%s vs %s" % (cat.CATEGORIES[a], cat.CATEGORIES[b])
            pairwise[key] = {"wilcoxon_p": float(pw), "cohens_d": d}
            print("    %-28s p = %.2e   d = %+.2f" % (key, pw, d))

    out[label] = {
        "means": {c: float(col.mean()) for c, col in zip(cat.CATEGORIES,
                                                         columns)},
        "sds": {c: float(col.std(ddof=1)) for c, col in zip(cat.CATEGORIES,
                                                            columns)},
        "ordering": order,
        "friedman_chi2": float(chi), "friedman_p": float(p),
        "pairwise": pairwise,
    }
    return order, pairwise


def main():
    quick = "--quick" in sys.argv
    with_coi = "--no-coi-check" not in sys.argv

    banner("Step 1  Co-modulation matrices and anatomical categories")
    angles = load("joint_angles_15hz.npz")
    stored = load("comodulation_matrices.npz")["xwt"]
    A, n_valid = angles["angles"], angles["n_samples"]
    n = 10 if quick else A.shape[0]
    print("  participants: %d%s" % (n, "  (quick run)" if quick else ""))

    iu = np.triu_indices(cat.N_ANGLES, 1)
    plain = np.zeros((n, 78))
    coned = np.zeros((n, 78)) if with_coi else None

    t0 = time.time()
    for s in range(n):
        p, c = matrices_for(A[s], int(n_valid[s]), with_coi)
        plain[s] = p
        if with_coi:
            coned[s] = c
        if (s + 1) % 10 == 0:
            print("    %d/%d  (%.0f s)" % (s + 1, n, time.time() - t0),
                  flush=True)

    # The recomputation has to land on the deposited matrices, otherwise
    # nothing below can be trusted.
    reference = np.asarray(stored[:n], float)[:, iu[0], iu[1]]
    rel = np.abs(plain - reference) / np.maximum(np.abs(reference), 1e-12)
    print("\n  agreement with the deposited matrices")
    print("    largest relative difference: %.2e" % rel.max())
    print("    median relative difference : %.2e" % np.median(rel))
    if rel.max() > 1e-5:
        print("    WARNING: the recomputation does not match the deposit")

    out = {"n_participants": int(n),
           "max_relative_difference_vs_deposit": float(rel.max())}

    order_plain, pairs_plain = category_report(
        plain, "session average over the full plane (as reported)", out)
    if with_coi:
        order_coned, pairs_coned = category_report(
            coned, "session average restricted to the cone of influence", out)
        same_order = order_plain == order_coned
        same_sig = all((pairs_plain[k]["wilcoxon_p"] < 0.05)
                       == (pairs_coned[k]["wilcoxon_p"] < 0.05)
                       for k in pairs_plain)
        drop = 100 * (np.mean([out[
            "session average restricted to the cone of influence"]["means"][c]
            / out["session average over the full plane (as reported)"][
                "means"][c] for c in cat.CATEGORIES]) - 1)
        print("\n  effect of restricting to the cone of influence")
        print("    every category shifts by %+.1f%%" % drop)
        print("    same category ordering            : %s"
              % ("yes" if same_order else "NO"))
        print("    same pattern of significance      : %s"
              % ("yes" if same_sig else "NO"))
        out["coi_sensitivity"] = {"mean_shift_pct": float(drop),
                                  "same_ordering": bool(same_order),
                                  "same_significance": bool(same_sig)}

    if not quick:
        print()
        compare("Intra-limb mean", out[
            "session average over the full plane (as reported)"]["means"][
            "Intra-limb"], PUBLISHED["intra_limb_mean"])
        compare("Homologous mean", out[
            "session average over the full plane (as reported)"]["means"][
            "Homologous"], PUBLISHED["homologous_mean"])
        compare("Cross-limb mean", out[
            "session average over the full plane (as reported)"]["means"][
            "Cross-limb"], PUBLISHED["cross_limb_mean"])
        compare("Head-body mean", out[
            "session average over the full plane (as reported)"]["means"][
            "Head-body"], PUBLISHED["head_body_mean"])
        compare("Friedman chi2", out[
            "session average over the full plane (as reported)"][
            "friedman_chi2"], PUBLISHED["friedman_chi2"])

    save_json("step1_comodulation.json", out)


if __name__ == "__main__":
    main()
