# -*- coding: utf-8 -*-
"""Convert a DeepLabCut .h5 output into the .csv the viewer reads.

Why this exists
---------------
DeepLabCut writes both formats, but a script that stitches the segments of a
long recording back together usually keeps only the .h5, because that is what
pandas works with. The viewer is an HTML page and a browser has no HDF5
reader; adding one would mean loading a multi-megabyte library to read a table
that in plain text reads itself.

So if what you have is a .h5, this turns it into a .csv without touching the
data: the same four header rows (scorer, individuals, bodyparts, coords) and
one row per frame.

Usage
-----
    python dlc_h5_to_csv.py MYVIDEO_superanimal_humanbody_....h5
    python dlc_h5_to_csv.py path/to/folder/

Writes the .csv next to each .h5. Nothing is overwritten without warning.
"""
import os
import sys

import pandas as pd


def convert(h5_path, decimals=2):
    target = os.path.splitext(h5_path)[0] + ".csv"
    if os.path.exists(target):
        print("  already there, left alone: %s" % os.path.basename(target))
        return target

    df = pd.read_hdf(h5_path)

    # DeepLabCut stores a multi-level column index: four levels for
    # multi-animal output (scorer, individuals, bodyparts, coords) and three
    # for a single individual. The viewer expects four, so the missing
    # individual level is added when it is not there.
    if df.columns.nlevels == 3:
        df.columns = pd.MultiIndex.from_tuples(
            [(a, "animal0", b, c) for a, b, c in df.columns],
            names=["scorer", "individuals", "bodyparts", "coords"])

    # DeepLabCut marks "no detection" by writing -1 in all three columns at
    # once, not by leaving the cell empty. It is turned into an empty cell,
    # which is what the viewer reads as missing; left as -1 it would be drawn
    # as a real point in the corner of the image.
    df = df.mask(df == -1)

    df.round(decimals).to_csv(target)
    print("  %s  ->  %s  (%d frames, %.1f MB)"
          % (os.path.basename(h5_path), os.path.basename(target), len(df),
             os.path.getsize(target) / 1e6))
    return target


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    target = sys.argv[1]

    if os.path.isdir(target):
        files = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if f.lower().endswith(".h5")]
        if not files:
            print("No .h5 files in %s" % target)
            return 1
    else:
        files = [target]

    print("converting %d file(s):" % len(files))
    for h5 in files:
        try:
            convert(h5)
        except Exception as e:                       # noqa: BLE001
            print("  *** %s: %s" % (os.path.basename(h5), e))
            return 1
    print("\nOpen the .csv with the 'Your data' button of the viewer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
