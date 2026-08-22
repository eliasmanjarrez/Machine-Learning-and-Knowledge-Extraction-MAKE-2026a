# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Shared plumbing for the step scripts: paths, loading and value checking.

Every step prints its own numbers next to the values printed in the article,
so that a reader can see at a glance whether the deposit reproduces the paper.
``compare`` is what does that, and it is deliberately strict: a mismatch is
printed as a mismatch rather than rounded away.
"""
from __future__ import annotations

import io
import json
import os
import sys


def _register_conda_dll_paths():
    """Make a conda environment's BLAS and LAPACK loadable on Windows.

    When a conda environment's interpreter is invoked by absolute path rather
    than activated, Windows does not search ``Library\\bin`` for dependent
    DLLs, and the first call that needs LAPACK aborts the process instead of
    raising. Registering the directories up front costs nothing elsewhere and
    is a no-op on Linux and macOS.
    """
    if os.name != "nt":
        return
    root = os.path.dirname(sys.executable)
    for sub in ("Library\\bin", "Library\\mingw-w64\\bin", "DLLs"):
        directory = os.path.join(root, sub)
        if not os.path.isdir(directory):
            continue
        try:
            os.add_dll_directory(directory)
        except (AttributeError, OSError):
            pass
        if directory not in os.environ.get("PATH", ""):
            os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]


_register_conda_dll_paths()

import numpy as np  # noqa: E402  (imported after the DLL directories)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, os.pardir, "data"))
RESULTS = os.path.normpath(os.path.join(HERE, os.pardir, "results"))

# Effective sampling interval of every population-level analysis: the signals
# are block-averaged from 60 fps to 15 Hz before the transform.
DT = 4.0 / 60.0

# Values as printed in the article, for the side-by-side check.
PUBLISHED = {
    "intra_limb_mean": 1839.7,
    "homologous_mean": 1812.4,
    "cross_limb_mean": 1433.1,
    "head_body_mean": 1117.7,
    "friedman_chi2": 126.6,
    "surrogate_excess_intra": 335.8,
    "surrogate_excess_homologous": 230.1,
    "surrogate_excess_cross": 37.8,
    "surrogate_excess_head": 5.9,
    "shared_three_mean": 2500.2,
    "shared_zero_mean": 1441.9,
    "zero_stratum_homologous": 1812.4,
    "zero_stratum_cross": 1485.8,
    "zero_stratum_head": 1117.7,
    "coherence_intra": 0.887,
    "coherence_homologous": 0.839,
    "coherence_cross": 0.830,
    "coherence_head": 0.824,
    "identification_forward_pct": 28.6,
    "identification_reverse_pct": 30.0,
    "identification_reduced_pct": 27.1,
    "anthropometry_pct": 48.6,
    "residual_5pc_pct": 24.3,
    "residual_10pc_pct": 17.1,
    "anthro_variance_5pc_pct": 72.0,
    "comod_removed_10pc_pct": 24.0,
    "icc_mean": 0.311,
    "consistency_within_z": 0.529,
    "consistency_between_z": 0.137,
    "micro_band_q": 0.022,
}


def banner(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def load(name):
    """Load one file from the deposited data directory."""
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        raise SystemExit(
            "missing %s\nRun the step scripts from the code directory of the "
            "deposit, with the data directory beside it." % path)
    if name.endswith(".npz"):
        return np.load(path, allow_pickle=True)
    import pandas as pd
    # The CSV files open with a citation line prefixed by '#', so that the
    # reference travels with the file if it is ever separated from this
    # deposit. Those lines are comments, not data.
    return pd.read_csv(path, comment="#")


def compare(label, obtained, published, tolerance=0.02):
    """Print a recomputed value beside the published one."""
    if published == 0:
        ok = abs(obtained) < tolerance
    else:
        ok = abs(obtained - published) / abs(published) <= tolerance
    print("  %-28s recomputed %12.4g   article %12.4g   %s"
          % (label, obtained, published, "match" if ok else "DIFFERS"))
    return ok


def save_json(name, payload):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    io.open(path, "w", encoding="utf-8").write(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    print("\n  written: %s" % os.path.relpath(path, HERE))
