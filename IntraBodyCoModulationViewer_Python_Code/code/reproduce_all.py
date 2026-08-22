# Intra-body joint-angle co-modulation: analysis pipeline.
#
# Zamora-Ursulo, M.A.; Flores, A.; Manjarrez, E. (2026)
# Interactive playback visualizer to analyze joint-angle co-modulation
# with a wavelet approach: application to pose-voice relationships
# during spontaneous conversation.
# Machine Learning and Knowledge Extraction.
#
# MIT License. Copyright (c) 2026 the authors.
"""Run every step and report whether the deposit reproduces the article.

    python reproduce_all.py           full run, roughly five minutes
    python reproduce_all.py --quick   skips the slowest recomputation

Each step prints its own numbers beside the values printed in the article. The
summary at the end counts how many of those checks matched, so a reader who
wants one answer rather than seven can read the last line.
"""
from __future__ import annotations

import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("step1_comodulation_and_categories.py",
     "co-modulation matrices and anatomical categories"),
    ("step2_surrogate_and_landmark_controls.py",
     "surrogate and shared-landmark controls"),
    ("step3_coherence_control.py", "wavelet coherence control"),
    ("step4_identification.py", "within-session identification"),
    ("step5_temporal_consistency.py", "within-session temporal consistency"),
    ("step6_sex_comparison.py", "sex comparison"),
]

CHECK = re.compile(r"recomputed\s+\S+\s+article\s+\S+\s+(match|DIFFERS)")


def main():
    quick = "--quick" in sys.argv
    matched = differed = 0
    failed = []

    for script, description in STEPS:
        print()
        print("#" * 74)
        print("# %s" % description)
        print("#" * 74)
        args = [sys.executable, os.path.join(HERE, script)]
        if quick and script.startswith("step1"):
            args.append("--quick")
        proc = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.stderr.strip():
            sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            failed.append(script)
            continue
        for m in CHECK.finditer(proc.stdout):
            if m.group(1) == "match":
                matched += 1
            else:
                differed += 1

    print()
    print("=" * 74)
    print("Summary")
    print("=" * 74)
    print("  values checked against the article: %d" % (matched + differed))
    print("  matched  : %d" % matched)
    print("  differed : %d" % differed)
    if failed:
        print("  scripts that failed to run: %s" % ", ".join(failed))
    if not failed and differed == 0:
        print()
        print("  Every reported value was reproduced from the deposited data.")
    return 1 if (failed or differed) else 0


if __name__ == "__main__":
    sys.exit(main())
