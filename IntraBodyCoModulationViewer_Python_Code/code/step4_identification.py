# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Step 4. Within-session identification, and what it actually rests on.

Splitting each recording into halves and matching a participant's first-half
co-modulation profile against every second-half profile identifies about three
in ten participants, far above the 1.4% chance rate. Both halves come from the
same recording, sharing camera, lighting, clothing, seating and adaptation
pass, so the result is identification within a session and not evidence of a
trait that would survive across days.

Three comparisons say what the accuracy is made of, and two of them qualify it.

Anthropometry alone identifies participants better than co-modulation does.
A vector of twelve median segment lengths and thirteen mean joint angles,
measured in the same two halves, reaches about half the sample. Body
proportions and mean posture are fixed within a session by camera geometry and
seating, so this is the clearest sign that the result is within-session
identifiability.

Accuracy also falls with the distance between the compared windows, so part of
what is measured is temporal proximity rather than a time-invariant signature.

What survives both is that identification persists once almost all of the
anthropometric variance has been regressed out, which means the co-modulation
profile carries individual information that body size and mean posture do not.

The script also repeats the identification on the reduced nine-angle set, as a
sensitivity analysis after excluding the four geometrically dependent distal
angles.

    python step4_identification.py
"""
from __future__ import annotations

import numpy as np
from scipy import linalg

from common import PUBLISHED, banner, compare, load, save_json
from intrabody import categories as cat
from intrabody.statistics import (identification_accuracy,
                                  identification_permutation_p,
                                  wilson_interval)


def profiles_from_windows(windows, indices=None):
    """Log10 profiles of the upper triangle, optionally on a subset of angles."""
    if indices is None:
        indices = list(range(cat.N_ANGLES))
    sub = np.asarray(windows, float)[:, :, indices][:, :, :, indices]
    iu = np.triu_indices(len(indices), 1)
    return np.log10(np.maximum(sub, 1e-6))[:, :, iu[0], iu[1]]


def residualize(profiles, anthropometry, n_components):
    """Remove from each co-modulation dimension what anthropometry predicts.

    The anthropometric features are standardized and reduced to their leading
    principal components before the regression. With 70 participants, using
    all 25 raw features as predictors would overfit and strip out legitimate
    variance along with the anthropometric part.

    Returns the residual profiles, the R-squared removed from each dimension,
    and the share of anthropometric variance the retained components span.
    """
    x = np.asarray(anthropometry, float)
    x = x - x.mean(0)
    x = x / (x.std(0) + 1e-12)

    # Eigendecomposition of the 25 x 25 correlation matrix, which is small.
    cov = np.einsum("ij,ik->jk", x, x) / (len(x) - 1)
    values, vectors = linalg.eigh(cov)
    order = np.argsort(-values)
    basis = vectors[:, order[:n_components]]
    captured = float(values[order[:n_components]].sum() / values.sum())

    design = np.column_stack([np.ones(len(x)), x @ basis])
    gram = np.einsum("ij,ik->jk", design, design)
    beta = linalg.solve(gram, np.einsum("ij,ik->jk", design, profiles),
                        assume_a="sym")
    fitted = design @ beta
    ss_res = ((profiles - fitted) ** 2).sum(0)
    ss_tot = ((profiles - profiles.mean(0)) ** 2).sum(0)
    return profiles - fitted, 1 - ss_res / (ss_tot + 1e-12), captured


def report(name, a, b, out, seed=0):
    fwd, rev, n, corr = identification_accuracy(a, b)
    lo, hi = wilson_interval(fwd, n)
    lo_r, hi_r = wilson_interval(rev, n)
    p = identification_permutation_p(corr, fwd, seed=seed)
    top5 = int(sum(i in np.argsort(-corr[i])[:5] for i in range(n)))
    print("    %-34s %2d/%d = %5.1f%%  [%.1f-%.1f]   reverse %5.1f%%   "
          "top-5 %5.1f%%   p %s"
          % (name, fwd, n, 100 * fwd / n, 100 * lo, 100 * hi,
             100 * rev / n, 100 * top5 / n,
             "< 1e-4" if p < 1e-4 else "%.4f" % p))
    out[name] = {"forward": fwd, "reverse": rev, "n": n,
                 "accuracy_pct": 100.0 * fwd / n,
                 "ci95_pct": [100 * lo, 100 * hi],
                 "reverse_ci95_pct": [100 * lo_r, 100 * hi_r],
                 "top5_pct": 100.0 * top5 / n,
                 "permutation_p": p}
    return out[name]


def main():
    banner("Step 4  Within-session identification")
    w = load("window_profiles.npz")["xwt"]
    n_windows = w.shape[1]
    half = n_windows // 2
    out = {}

    print("\n  identification accuracy (chance = %.1f%%)" % (100.0 / len(w)))
    full = profiles_from_windows(w)
    a = full[:, :half].mean(axis=1)
    b = full[:, -half:].mean(axis=1)
    report("co-modulation, 13 angles", a, b, out)

    reduced = profiles_from_windows(w, cat.REDUCED_INDICES)
    report("co-modulation, 9 angles (control)",
           reduced[:, :half].mean(axis=1),
           reduced[:, -half:].mean(axis=1), out)

    # Anthropometry, the baseline that beats co-modulation.
    anth = load("anthropometry.csv")
    cols = [c for c in anth.columns if c not in ("participant", "half")]
    first = anth[anth["half"] == "first"][cols].to_numpy(float)
    second = anth[anth["half"] == "second"][cols].to_numpy(float)
    keep = ~(np.isnan(first).any(axis=0) | np.isnan(second).any(axis=0))
    report("anthropometry alone", first[:, keep], second[:, keep], out)

    # How much of the co-modulation profile anthropometry explains away.
    print("\n  identification after removing anthropometric components")
    for n_pc in (3, 5, 10):
        resid_a, r2_a, captured = residualize(a, first[:, keep], n_pc)
        resid_b, r2_b, _ = residualize(b, second[:, keep], n_pc)
        r = report("residual after %2d components" % n_pc,
                   resid_a, resid_b, out)
        r["anthropometric_variance_captured_pct"] = float(100 * captured)
        r["comodulation_variance_removed_pct"] = float(
            100 * np.mean(np.r_[r2_a, r2_b]))
        print("      components span %.0f%% of the anthropometric variance "
              "and remove %.1f%% of the co-modulation variance"
              % (r["anthropometric_variance_captured_pct"],
                 r["comodulation_variance_removed_pct"]))

    # Accuracy against the amount of data and the separation of the windows.
    print("\n  accuracy against the amount of data compared")
    out["segment_length"] = {}
    for k in (1, 2, 3):
        fwd, rev, n, _ = identification_accuracy(full[:, :k].mean(axis=1),
                                                 full[:, -k:].mean(axis=1))
        out["segment_length"]["%d_min_per_half" % (2 * k)] = 100.0 * fwd / n
        print("    %2d minutes per half   %5.1f%%" % (2 * k, 100.0 * fwd / n))

    print("\n  accuracy against the separation between single windows")
    out["window_separation"] = {}
    for gap in (1, 3, n_windows - 1):
        fwd, rev, n, _ = identification_accuracy(full[:, 0], full[:, gap])
        out["window_separation"]["%d_min_apart" % (2 * gap)] = 100.0 * fwd / n
        print("    %2d minutes apart      %5.1f%%" % (2 * gap, 100.0 * fwd / n))

    print()
    compare("identification 13 angles",
            out["co-modulation, 13 angles"]["accuracy_pct"],
            PUBLISHED["identification_forward_pct"])
    compare("identification 9 angles",
            out["co-modulation, 9 angles (control)"]["accuracy_pct"],
            PUBLISHED["identification_reduced_pct"])
    compare("anthropometry alone",
            out["anthropometry alone"]["accuracy_pct"],
            PUBLISHED["anthropometry_pct"])
    compare("residual, 5 components",
            out["residual after  5 components"]["accuracy_pct"],
            PUBLISHED["residual_5pc_pct"])
    compare("residual, 10 components",
            out["residual after 10 components"]["accuracy_pct"],
            PUBLISHED["residual_10pc_pct"])
    compare("anthro variance, 5 PCs",
            out["residual after  5 components"][
                "anthropometric_variance_captured_pct"],
            PUBLISHED["anthro_variance_5pc_pct"])
    compare("co-mod variance removed, 10 PCs",
            out["residual after 10 components"][
                "comodulation_variance_removed_pct"],
            PUBLISHED["comod_removed_10pc_pct"])

    save_json("step4_identification.json", out)


if __name__ == "__main__":
    main()
