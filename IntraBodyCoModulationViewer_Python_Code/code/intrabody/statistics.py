# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Statistical routines used in the article, in the form the article uses them.

Three choices here are deliberate and worth stating, because they are the
points on which the reported inference turns.

The participant is the unit. Every comparison collapses a participant's many
joint pairs into one value per condition before testing. Pooling the pairs
themselves would treat one person's 78 correlated measurements as 78
independent observations and inflate significance by orders of magnitude.

Comparisons between women and men use Welch's t-test rather than the
equal-variance form. In the micro-movement band the standard deviation is 3.6
times larger in women than in men, and assuming a common variance there gives
a materially different answer.

Proportions carry Wilson intervals rather than normal approximations, because
identification accuracies near 25% with n = 70 sit where the two disagree.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def benjamini_hochberg(pvalues):
    """FDR-adjusted q-values, returned in the order of the input."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate((p * n / ranks)[order][::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def wilson_interval(successes, trials, z=1.96):
    """Wilson score interval for a proportion.

    Returns (low, high) as proportions. With 20 correct identifications out of
    70 this gives 0.193 to 0.401, the interval reported in the article.
    """
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials
                         + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher_z(r):
    """Fisher transformation, with r clipped away from the singularities."""
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    return np.arctanh(r)


def paired_cohens_d(a, b):
    """Cohen's d for paired samples: mean difference over its own SD."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(d.mean() / d.std(ddof=1))


def hedges_g(group_a, group_b):
    """Hedges' g for two independent groups, with the small-sample correction.

    Returned as group_a minus group_b, so a positive value means group_a is
    larger.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                     / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / pooled
    return float(d * (1 - 3.0 / (4 * (n1 + n2) - 9)))


def welch_comparison(group_a, group_b):
    """Welch's t-test with a 95% confidence interval on the difference.

    Returns a dict with the difference (group_a minus group_b), its interval,
    the t statistic, the Welch-Satterthwaite degrees of freedom, the p-value,
    Hedges' g and the distribution-free Mann-Whitney p-value alongside it.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")

    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                    + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    crit = stats.t.ppf(0.975, df)
    return {"difference": float(diff),
            "ci95": (float(diff - crit * se), float(diff + crit * se)),
            "t": float(t), "df": float(df), "p": float(p),
            "mannwhitney_p": float(p_u),
            "hedges_g": hedges_g(a, b)}


def zscore_rows(matrix):
    """Standardize each row to zero mean and unit variance."""
    x = np.asarray(matrix, dtype=float)
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-12)


def identification_accuracy(profiles_a, profiles_b):
    """Nearest-neighbour identification between two sets of profiles.

    Each row is one participant's profile. Rows are standardized, correlated
    across the two sets, and a participant counts as identified when their own
    profile is the most similar one.

    Returns (n_correct_forward, n_correct_reverse, n, correlation_matrix).
    """
    a = zscore_rows(profiles_a)
    b = zscore_rows(profiles_b)
    c = np.einsum("ik,jk->ij", a, b) / a.shape[1]
    n = len(a)
    forward = int((c.argmax(axis=1) == np.arange(n)).sum())
    reverse = int((c.T.argmax(axis=1) == np.arange(n)).sum())
    return forward, reverse, n, c


def identification_permutation_p(corr, observed, n_iter=10000, seed=0):
    """Permutation p-value for an identification accuracy.

    The columns of the correlation matrix are shuffled, which destroys the
    correspondence between the two halves while leaving the profiles intact.
    """
    rng = np.random.default_rng(seed)
    n = corr.shape[0]
    hits = 0
    for _ in range(n_iter):
        cols = rng.permutation(n)
        acc = int((corr[:, cols].argmax(axis=1) == np.arange(n)).sum())
        if acc >= observed:
            hits += 1
    return (1.0 + hits) / (n_iter + 1.0)
