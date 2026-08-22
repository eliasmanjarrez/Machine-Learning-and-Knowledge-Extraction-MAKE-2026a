# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Intra-body joint-angle co-modulation: analysis library.

Companion code for

    Zamora-Ursulo, M.A., Flores, A., Manjarrez, E.
    Interactive playback visualizer to analyse joint-angle co-modulation with
    a wavelet approach: application to pose-voice relationships during
    spontaneous conversation.


The library is deliberately small. It contains the wavelet core that produced
every matrix in the article, the anatomical bookkeeping that assigns joint
pairs to categories, and the handful of statistical routines the paper
reports. Everything else lives in the numbered `step*.py` scripts, one per
analysis, so that each reported number can be traced to the lines that
produce it.
"""

__version__ = "1.0.0"
__all__ = ["wavelet", "categories", "statistics"]
