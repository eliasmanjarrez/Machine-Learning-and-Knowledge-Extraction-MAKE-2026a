# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Step 6. Comparison between women and men, one observation per participant.

Each participant contributes a single value to each of eight measures: overall
co-modulation, the four anatomical categories and the three frequency bands.
Comparing the pairs themselves would treat one person's 78 correlated
measurements as independent observations, so the analysis collapses to the
participant first.

Two details decide the outcome and are worth stating plainly.

The test is Welch's, not the equal-variance form. In the micro-movement band
the standard deviation is 3.6 times larger in women than in men, and pooling
those variances gives a materially different answer.

The micro band is additionally tested on log10 values, which is what the
heteroscedasticity calls for. That is the primary test for that band, and the
untransformed and rank-based results are printed beside it because the band
sits close to the correction threshold and readers should see all three.

The result: no significant difference in overall or category-level
co-modulation, and a difference in the micro band with women above men. That
band is the narrowest of the three and the one most attenuated by the 4-frame
block average and the 5-frame median filter applied during conditioning, so
the difference is reported and not interpreted as a property of
micro-movement.

    python step6_sex_comparison.py
"""
from __future__ import annotations

import numpy as np

from common import PUBLISHED, banner, compare, load, save_json
from intrabody import categories as cat
from intrabody.statistics import benjamini_hochberg, welch_comparison

MEASURES = ["Global", "Intra-limb", "Homologous", "Cross-limb", "Head-body",
            "Postural", "Gesture", "Micro"]


def main():
    banner("Step 6  Sex comparison at the participant level")
    xwt = np.asarray(load("comodulation_matrices.npz")["xwt"], float)
    bands = load("band_power.csv")
    people = load("participants.csv")

    iu = np.triu_indices(cat.N_ANGLES, 1)
    pairs = xwt[:, iu[0], iu[1]]
    idx = cat.category_indices()

    values = {"Global": pairs.mean(axis=1)}
    for c in cat.CATEGORIES:
        values[c] = pairs[:, idx[c]].mean(axis=1)
    values["Postural"] = bands["postural_below_0p5Hz"].to_numpy(float)
    values["Gesture"] = bands["gesture_0p5_to_5Hz"].to_numpy(float)
    values["Micro"] = bands["micro_5_to_7p162Hz"].to_numpy(float)

    sex = people["sex"].to_numpy()
    women, men = sex == "W", sex == "M"
    print("  %d women, %d men" % (women.sum(), men.sum()))

    rows, primary_p = [], []
    for name in MEASURES:
        v = values[name]
        r = welch_comparison(v[men], v[women])       # men minus women
        r["measure"] = name
        r["mean_women"] = float(v[women].mean())
        r["sd_women"] = float(v[women].std(ddof=1))
        r["mean_men"] = float(v[men].mean())
        r["sd_men"] = float(v[men].std(ddof=1))
        if name == "Micro":
            lg = np.log10(np.maximum(v, 1e-9))
            r["log10"] = welch_comparison(lg[men], lg[women])
            r["sd_ratio_women_over_men"] = r["sd_women"] / r["sd_men"]
            r["primary_test"] = "Welch on log10 values"
            r["p_primary"] = r["log10"]["p"]
        else:
            r["primary_test"] = "Welch on untransformed values"
            r["p_primary"] = r["p"]
        rows.append(r)
        primary_p.append(r["p_primary"])

    for r, q in zip(rows, benjamini_hochberg(primary_p)):
        r["q_fdr"] = float(q)

    print("\n  measure       women            men              difference"
          "        g       p        q")
    for r in rows:
        print("  %-12s %8.1f+/-%-7.1f %8.1f+/-%-7.1f %9.2f  %+.2f  %.4f  "
              "%.4f%s"
              % (r["measure"], r["mean_women"], r["sd_women"],
                 r["mean_men"], r["sd_men"], r["difference"],
                 r["hedges_g"], r["p_primary"], r["q_fdr"],
                 "  <-- q < 0.05" if r["q_fdr"] < 0.05 else ""))

    micro = rows[-1]
    print("\n  micro band, all three tests")
    print("    Welch on log10 (primary)  p = %.4f" % micro["log10"]["p"])
    print("    Welch untransformed       p = %.4f" % micro["p"])
    print("    Mann-Whitney              p = %.4f" % micro["mannwhitney_p"])
    print("    standard deviation ratio, women over men: %.1f"
          % micro["sd_ratio_women_over_men"])
    print("    Reported, not interpreted: this band is attenuated by the")
    print("    conditioning filters and is the least reliable of the three.")

    significant = [r["measure"] for r in rows if r["q_fdr"] < 0.05]
    out = {"n_women": int(women.sum()), "n_men": int(men.sum()),
           "measures": rows, "significant_after_fdr": significant}

    print()
    compare("micro band q", micro["q_fdr"], PUBLISHED["micro_band_q"], 0.1)
    save_json("step6_sex_comparison.json", out)


if __name__ == "__main__":
    main()
